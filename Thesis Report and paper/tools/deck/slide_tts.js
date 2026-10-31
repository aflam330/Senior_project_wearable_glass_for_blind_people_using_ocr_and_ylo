  // ================================================================ 18b. text-to-speech
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Reading and voice" });
  s.addText("Text-to-Speech: Spoken Answers", { placeholder: "title" });
  circleIcon(s, "speak", 0.6, 1.45, 0.8, HEX.accent1, "Speech");
  s.addText("How the glass speaks", { x: 1.6, y: 1.6, w: 5.0, h: 0.5, fontFace: "Arial", fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Speech heading" });
  s.addText([
    { text: "Every answer becomes one Bangla or English sentence; the language is found per sentence", options: { bullet: true, breakLine: true } },
    { text: "Piper neural voices " + cite("piper") + ", offline; eSpeak NG " + cite("espeak") + " as a fallback", options: { bullet: true, breakLine: true } },
    { text: "Each sentence is made in full, then played; the next one is prepared meanwhile, so the sound does not break when the Pi is busy", options: { bullet: true, breakLine: true } },
    { text: "Sentences stay whole (they used to be cut at every comma and digit)", options: { bullet: true } },
  ], { x: 0.6, y: 2.4, w: 6.0, h: 2.9, fontSize: 16, color: C.text1, isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: 8, objectName: "Speech list" });
  stat(s, 0.6, 5.2, 2.9, "14 ms", "to make a short Bangla sentence on the Pi (eSpeak NG)", "Speech Pi");
  stat(s, 3.7, 5.2, 2.9, "0.2–0.5 s", "for a Piper Bangla sentence (laptop)", "Speech Piper");
  s.addText("Is the Bangla voice understood?", { x: 7.0, y: 1.5, w: 5.7, h: 0.5, fontFace: "Arial", fontSize: 16, bold: true, color: C.text2, isTextBox: true, margin: 0, objectName: "Listener heading" });
  const heard = [
    ["পাঁচশত টাকার নোট সম্ভবত আসল", "পাঁচশত টাকার নোট সম্ভবত আসল", "D9F2E3"],
    ["ধূমপান নিষেধ", "ধূমপান নিষেধ", "D9F2E3"],
    ["এক হাজার টাকার নোট", "1000 টাকা নোট", "FFF4D6"],
    ["কোনো লেখা পাওয়া যায়নি", "কোন লেখা পাওয়া যায়নি", "FFF4D6"],
    ["সামনে আছে চেয়ার টেবিল", "স্বামী আছে যে টেবিল", "FBE9EB"],
    ["নোটটি আলোর সামনে ধরুন", "নটি আলু সামনে ধরুন", "FBE9EB"],
  ];
  const thd = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, fontFace: "Arial", fontSize: 13 } });
  s.addTable([[thd("The glass said"), thd("The listener heard")],
    ...heard.map(([a, b, col]) => [a, b].map((t) => ({ text: t, options: { fontFace: BN, fontSize: 14, color: HEX.dk1, fill: { color: col } } })))],
    { x: 7.0, y: 2.05, w: 5.7, colW: [2.85, 2.85], rowH: 0.5, border: { type: "solid", color: "FFFFFF", pt: 1.5 }, valign: "middle", margin: 0.06, objectName: "Listener table" });
  s.addText("2 exact, 2 small differences, 2 with wrong words. A machine listener, not people.",
    { x: 7.0, y: 5.65, w: 5.7, h: 0.6, fontSize: 13, italic: true, color: C.accent2, isTextBox: true, margin: 0, valign: "top", objectName: "Listener note" });
  sources(s, `Voices ${cite("piper", "espeak")}. Results: savior_glass/results/speech_piper_bn_intelligibility.json (Google Web Speech, bn-BD, as the listener); Pi time from benchmark_report/.`);
  notes(s, "Every answer of the glass becomes one sentence in Bangla or English, and the language is found sentence by sentence. The glass speaks with Piper neural voices, offline, with eSpeak NG as a fallback. Each sentence is made in full and then played, and the next one is prepared while the current one is heard, so the sound no longer breaks when the Pi is busy reading text; sentences also stay whole instead of being cut at every comma and digit. Making a short Bangla sentence takes 14 milliseconds on the Pi with eSpeak NG, and 0.2 to 0.5 seconds with the Piper voice on the laptop. To check whether the Bangla voice can be understood, six sentences were given to an online Bangla speech recogniser: two came back exactly, two with small differences, and two with wrong words. This is a machine listener, not people.");

