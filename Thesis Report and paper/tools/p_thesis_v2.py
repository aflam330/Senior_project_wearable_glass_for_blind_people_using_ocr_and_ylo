"""Thesis: the second prototype (26 October 2026) — figure, text, hardware table pictures, comparison thumbnails."""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "thesis report" / "My report"


def patch(name, pairs):
    p = D / name
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert s.count(a) >= 1, (name, a[:80])
        s = s.replace(a, b)
    assert not any(ord(c) < 32 and c not in "\n\t\r" for c in s), name
    p.write_text(s, encoding="utf-8")


patch("methodology.tex", [
    (r"""    \caption{First prototype of SARA: the camera module above the bridge of a 3D-printed frame (left), the Raspberry Pi~5 with heat sink and fan tied to the right temple (centre), and the ribbon cable between camera and board (right)}
    \label{fig:prototype}
\end{figure}
""",
     r"""    \caption{First prototype of SARA: the camera module above the bridge of a 3D-printed frame (left), the Raspberry Pi~5 with heat sink and fan tied to the right temple (centre), and the ribbon cable between camera and board (right)}
    \label{fig:prototype}
\end{figure}

The second prototype (Figure~\ref{fig:prototype2}) answers these problems while keeping everything on the head. The camera now sits in a small closed housing on the front of the frame, with a blue LED beside it. The Raspberry Pi~5 moved from the temple into a 3D-printed enclosure with ventilation slots at the back of the head, so that the board balances the frame instead of pulling it to one side and its fan blows away from the face. A strap joins the frame to the enclosure and carries the wires. The five buttons sit in a row on a small perforated board along the temple, where the fingers find them without looking, and an open-ear bone-conduction headphone delivers the speech while leaving the ears free for traffic and voices. Power still comes from a power bank through a cable.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.62\textwidth]{sara_v2_collage.jpg}
    \caption{Second prototype of SARA worn by a team member: the camera housing and LED at the front of the frame (top), the five-button board along the temple and the bone-conduction headphone (bottom left and top right), and the ventilated enclosure carrying the Raspberry Pi~5 at the back of the head (bottom right)}
    \label{fig:prototype2}
\end{figure}
"""),
    # hardware table: our own pictures for buttons, headphone and frame; a row for the enclosure
    (r"""\vspace{2pt}\includegraphics[width=2.4cm,height=1.9cm,keepaspectratio]{hw_buttons.jpg}\vspace{1pt} & Push-buttons (5) & Press-to-click buttons connected to Pi pins 17, 27, 24, 22 and 23 &""",
     r"""\vspace{2pt}\includegraphics[width=2.4cm,height=1.9cm,keepaspectratio]{v2_buttons.jpg}\vspace{1pt} & Push-buttons (5) & Press-to-click buttons on a small perforated board along the temple, connected to Pi pins 17, 27, 24, 22 and 23 &"""),
    (r"""\vspace{2pt}\includegraphics[width=2.4cm,height=1.9cm,keepaspectratio]{hw_earphone.jpg}\vspace{1pt} & Earphone / speaker & Wired 3.5 mm or USB earphone, or a small speaker & Speaks every result to the user \\ \hline""",
     r"""\vspace{2pt}\includegraphics[width=2.4cm,height=1.9cm,keepaspectratio]{v2_headphone.jpg}\vspace{1pt} & Bone-conduction headphone & Open-ear headphone that passes sound through the cheekbones; any earphone or small speaker also works & Speaks every result while the ears stay free for traffic and voices \\ \hline"""),
    (r"""\vspace{2pt}\includegraphics[width=2.4cm,height=1.9cm,keepaspectratio]{hw_frame.jpg}\vspace{1pt} & Spectacle frame & 3D-printed frame with the camera above the nose bridge & Holds the camera where the user looks, so the glass is worn like ordinary spectacles \\ \hline""",
     r"""\vspace{2pt}\includegraphics[width=2.4cm,height=1.9cm,keepaspectratio]{v2_frame.jpg}\vspace{1pt} & Spectacle frame & 3D-printed frame with the camera housing and a blue LED at the front, held by a head strap & Holds the camera where the user looks, so the glass is worn like ordinary spectacles \\ \hline
\vspace{2pt}\includegraphics[width=2.4cm,height=1.9cm,keepaspectratio]{v2_enclosure.jpg}\vspace{1pt} & Rear enclosure & 3D-printed case with ventilation slots at the back of the head & Carries the Raspberry Pi~5, balances the frame and keeps the board's heat away from the face \\ \hline"""),
    (r"""tactile push-buttons and vibration motor, by Raimond Spekking, CC BY-SA 4.0; earphones, by Berthold Werner, CC BY-SA 3.0; USB power bank, by Pinguinguy, CC0. Spectacle frame: our first prototype.""",
     r"""vibration motor, by Raimond Spekking, CC BY-SA 4.0; USB power bank, by Pinguinguy, CC0. Push-buttons, headphone, spectacle frame and enclosure: our second prototype."""),
])

patch("implementation.tex", [
    (r"""The camera module is mounted above the bridge of the spectacle frame so that it points where the user's face points. In the first prototype the Raspberry Pi~5 is tied to the right temple of the frame and the ribbon cable loops over the head from the camera to the board (Figure~\ref{fig:prototype}); Figure~\ref{fig:worn} shows it being worn.""",
     r"""The camera module is mounted at the front of the spectacle frame so that it points where the user's face points. In the first prototype the Raspberry Pi~5 is tied to the right temple of the frame and the ribbon cable loops over the head from the camera to the board (Figure~\ref{fig:prototype}); Figure~\ref{fig:worn} shows it being worn. In the second prototype (Figure~\ref{fig:prototype2}) the camera sits in a closed housing with an LED beside it, the board rides in a ventilated enclosure at the back of the head, a strap joins the two and carries the wires, the five buttons are on a board along the temple, and a bone-conduction headphone gives the speech."""),
])

for f in ("LiteratureReview.tex",):
    patch(f, [(r"""\IfFileExists{DIagram fig tab/savior_glass_prototype.jpg}{\vspace{2pt}\includegraphics[width=2.5cm]{savior_glass_prototype.jpg}\vspace{1pt}}""",
               r"""\IfFileExists{DIagram fig tab/sara_v2_thumb.jpg}{\vspace{2pt}\includegraphics[width=2.5cm]{sara_v2_thumb.jpg}\vspace{1pt}}"""),
              (r"""\IfFileExists{DIagram fig tab/savior_glass_prototype.jpg}{\vspace{2pt}\includegraphics[width=2.1cm]{savior_glass_prototype.jpg}\vspace{1pt}}""",
               r"""\IfFileExists{DIagram fig tab/sara_v2_thumb.jpg}{\vspace{2pt}\includegraphics[width=2.1cm]{sara_v2_thumb.jpg}\vspace{1pt}}""")])
print("thesis v2 ok")
