# SPDX-License-Identifier: MIT
"""Synthetic NIST Color FERET ground-truth tree for the tests.

Only the file layout and the record formats follow the NIST distribution; the
subject numbers (09001, 09002 lie outside the NIST range), dates, attributes and
coordinates are made up.
"""

from __future__ import annotations

SUBJECT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Subjects>
<Subject id="cfrS{sid}">
  <Gender value="{gender}" source="Retrospectively"/>
  <YOB value="{yob}" source="Retrospectively"/>
  <Race value="{race}" source="Retrospectively"/>
</Subject>
</Subjects>
"""
RECORDING_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Recordings>
<Recording id="cfrR00001">
  <URL root="Disc1" relative="data/images/{sid}/{img}.ppm.bz2"/>
  <CaptureDate>{date}</CaptureDate>
  <Subject id="cfrS{sid}">
   <Application>
    <Face>
     <Pose name="{pose}" yaw="{yaw}" pitch="0" roll="0"/>
     <Wearing glasses="{glasses}"/>
     <Hair beard="{beard}" mustache="{mustache}" source="Retrospectively"/>
     <LeftEye x="100" y="100"/>
    </Face>
   </Application>
  </Subject>
 </Recording>
</Recordings>
"""
SUBJECTS = {
    "09002": dict(gender="Female", yob="1965", race="Asian-Middle-Eastern"),
    "09001": dict(gender="Male", yob="1950", race="Black-or-African-American"),
}
RECORDINGS = [
    ("09002", "09002_910315_hl", "03/15/1991", "hl", "67.5", "No", "No", "No"),
    ("09001", "09001_900101_fb_a", "01/01/1990", "fb", "0", "No", "No", "Yes"),
    ("09001", "09001_900101_fa_a", "01/01/1990", "fa", "0", "Yes", "Yes", "No"),
]


def write_tree(root, dvd_of=lambda sid: "dvd1" if sid == "09001" else "dvd2"):
    for sid, s in SUBJECTS.items():
        for fmt in ("xml", "name_value"):
            d = root / "colorferet" / dvd_of(sid) / "data" / "ground_truths" / fmt / sid
            d.mkdir(parents=True, exist_ok=True)
        base = root / "colorferet" / dvd_of(sid) / "data" / "ground_truths"
        (base / "xml" / sid / f"{sid}.xml").write_text(SUBJECT_XML.format(sid=sid, **s))
        (base / "name_value" / sid / f"{sid}.txt").write_text(
            f"id=cfrS{sid}\ngender={s['gender']}\nyob={s['yob']}\nrace={s['race']}\n"
        )
    for sid, img, date, pose, yaw, glasses, beard, mustache in RECORDINGS:
        base = root / "colorferet" / dvd_of(sid) / "data" / "ground_truths"
        (base / "xml" / sid / f"{img}.xml").write_text(
            RECORDING_XML.format(
                sid=sid,
                img=img,
                date=date,
                pose=pose,
                yaw=yaw,
                glasses=glasses,
                beard=beard,
                mustache=mustache,
            )
        )
        (base / "name_value" / sid / f"{img}.txt").write_text(
            f"capture_date={date}\npose={pose}\nyaw={yaw}\npitch=0\nroll=0\n"
            f"glasses={glasses}\nbeard={beard}\nmustache={mustache}\n"
            "left_eye_coordinates=100 100\n"
        )
    # distractors: the global XML files and the grey FERET ground truth
    (
        root / "colorferet" / "dvd1" / "data" / "ground_truths" / "xml" / "subjects.xml"
    ).write_text("<x/>")
    grey = (
        root
        / "colorferet"
        / "dvd2"
        / "gray_feret_cd1"
        / "data"
        / "ground_truths"
        / "xml"
        / "09001"
    )
    grey.mkdir(parents=True)
    (grey / "09001_900101_fa.xml").write_text("<broken")
