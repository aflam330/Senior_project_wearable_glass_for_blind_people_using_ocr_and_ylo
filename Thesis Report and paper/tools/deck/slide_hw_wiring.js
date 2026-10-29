  // ================================================================ 8b. hardware design (wiring diagram)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Solution" });
  s.addText("Hardware Design", { placeholder: "title" });
  // a wire through the points [[x, y], ...], drawn as horizontal and vertical segments
  const wire = (pts, color, name, width = 2.25, dash) => {
    for (let k = 0; k + 1 < pts.length; k++) {
      const [x1, y1] = pts[k], [x2, y2] = pts[k + 1];
      s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.01), h: Math.max(Math.abs(y2 - y1), 0.01),
        line: { color, width, ...(dash ? { dashType: dash } : {}) }, objectName: name + " " + (k + 1) });
    }
  };
  const wlab = (t, x, y, w, align = "left") => s.addText(t, { x, y, w, h: 0.22, fontSize: 10, color: HEX.dk1, align, isTextBox: true, margin: 0, objectName: "Wire label: " + t });
  const comp = (file, label, x, y, w, h, name) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: HEX.accent6, width: 1 }, objectName: name + " card" });
    s.addImage({ path: FIG + file, x: x + 0.08, y: y + 0.08, w: w - 0.16, h: h - 0.55, sizing: { type: "contain", w: w - 0.16, h: h - 0.55 }, objectName: name + " photo" });
    s.addText(label, { x: x + 0.05, y: y + h - 0.45, w: w - 0.1, h: 0.4, fontSize: 11, bold: true, color: HEX.dk2, align: "center", valign: "middle", isTextBox: true, margin: 0, objectName: name + " label" });
  };
  const RED = "D7263D", BLACK = "1B2430", USB = "5B6777";
  // the board
  s.addImage({ path: FIG + "hw_pi5.jpg", x: 4.85, y: 2.75, w: 3.6, h: 2.7, objectName: "Raspberry Pi 5" });
  s.addText("Raspberry Pi 5, 8 GB " + cite("rpi5") + ", in the rear enclosure", { x: 4.35, y: 5.48, w: 4.6, h: 0.3, fontSize: 13, bold: true, color: HEX.dk2, align: "center", isTextBox: true, margin: 0, objectName: "Pi label" });
  // left: camera, headphone, power bank
  comp("hw_webcam.jpg", "USB webcam", 0.6, 1.35, 2.0, 1.55, "Camera");
  wire([[2.6, 2.1], [3.9, 2.1], [3.9, 3.6], [4.85, 3.6]], USB, "Webcam USB cable", 4);
  wlab("USB cable", 2.7, 1.85, 1.2);
  comp("v2_headphone.jpg", "Bone-conduction headphone", 0.6, 3.1, 2.0, 1.45, "Headphone");
  wire([[2.6, 3.85], [3.55, 3.85], [3.55, 4.3], [4.85, 4.3]], "7B4FA0", "Audio", 2.25, "dash");
  wlab("audio out", 2.7, 3.6, 1.0);
  comp("hw_powerbank.jpg", "Power bank", 0.6, 4.75, 2.0, 1.45, "Power bank");
  wire([[2.6, 5.3], [4.3, 5.3], [4.3, 5.0], [4.85, 5.0]], RED, "Power", 3.5);
  wlab("USB-C, 5 V", 2.7, 5.05, 1.4);
  // right: buttons, vibration motor, frame with the UV lamp
  comp("v2_buttons.jpg", "Five push-buttons", 10.7, 1.35, 2.0, 1.85, "Buttons");
  const pins = [["MODE · BCM 17", "1F5F8B"], ["ACTION · BCM 27", "2E9E5B"], ["READ · BCM 24", "E6A817"], ["VOL+ · BCM 22", "4A87B5"], ["VOL− · BCM 23", "B5651D"]];
  pins.forEach(([t, col], k) => {
    const y = 1.5 + k * 0.22, x = 7.0 + k * 0.2;
    wire([[10.7, y], [x, y], [x, 2.85]], col, t + " wire", 2);
    wlab(t, 8.95, y - 0.21, 1.7, "right");
  });
  wire([[10.7, 2.62], [6.75, 2.62], [6.75, 2.85]], BLACK, "Ground", 2);
  wlab("GND", 9.6, 2.4, 1.05, "right");
  comp("hw_motor.jpg", "Vibration motor", 10.7, 3.4, 2.0, 1.35, "Motor");
  wire([[10.7, 3.95], [9.3, 3.95], [9.3, 2.74], [8.2, 2.74], [8.2, 2.85]], "C2185B", "Motor signal", 2);
  wire([[10.7, 4.2], [9.55, 4.2], [9.55, 2.62]], BLACK, "Motor ground", 2);
  s.addShape(pres.shapes.OVAL, { x: 9.5, y: 2.57, w: 0.1, h: 0.1, fill: { color: BLACK }, line: { color: BLACK }, objectName: "Ground junction" });
  wlab("BCM 13 (PWM)", 9.6, 3.7, 1.05, "right");
  wire([[8.45, 4.75], [9.35, 4.75]], USB, "USB", 4);
  s.addText("USB microphone (optional)", { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.05, x: 9.35, y: 4.5, w: 1.2, h: 0.5, fill: { color: "FFFFFF" }, line: { color: HEX.accent6 },
    fontSize: 10, color: HEX.dk1, align: "center", valign: "middle", isTextBox: true, margin: 0.03, objectName: "Microphone" });
  comp("v2_frame.jpg", "Frame + UV-A lamp", 10.7, 4.95, 2.0, 1.3, "Frame");
  s.addText("Lamp 360–395 nm, switched by hand; not wired to the Pi", { x: 8.75, y: 5.3, w: 1.85, h: 0.65, fontSize: 10, italic: true, color: HEX.dk1, align: "right", valign: "middle", isTextBox: true, margin: 0, objectName: "Lamp note" });
  // legend
  const legend = [[RED, "5 V power", 3.5], [USB, "USB", 4], [HEX.accent1, "GPIO input, to GND with pull-up", 2], [BLACK, "ground", 2]];
  legend.forEach(([col, t, wd], k) => {
    const x = [3.0, 4.55, 6.3, 9.1][k];
    s.addShape(pres.shapes.LINE, { x, y: 6.0, w: 0.35, h: 0, line: { color: col, width: wd }, objectName: "Legend line " + (k + 1) });
    s.addText(t, { x: x + 0.42, y: 5.88, w: k === 2 ? 2.4 : 1.3, h: 0.25, fontSize: 10, color: HEX.dk1, isTextBox: true, margin: 0, objectName: "Legend " + (k + 1) });
  });
  s.addText("Schematic, not to scale; pin numbers are those set in the code.", { x: 3.0, y: 6.2, w: 6.0, h: 0.25, fontSize: 10, italic: true, color: C.accent5, isTextBox: true, margin: 0, objectName: "Schematic note" });
  sources(s, "Pictures (Wikimedia Commons): Raspberry Pi 5 by SimonWaldherr, CC BY 4.0; USB webcam by WrS.tm.pl, CC0; vibration motor by Raimond Spekking, CC BY-SA 4.0; power bank by Pinguinguy, CC0. Buttons, headphone and frame: our prototype.");
  notes(s, "This is how the second prototype is wired. The Raspberry Pi 5 sits in the rear enclosure and runs every model. The camera is a USB webcam plugged into a USB port of the Pi. The five push-buttons each connect one GPIO pin to ground, BCM 17 for MODE, 27 for ACTION, 24 for READ, 22 for volume up and 23 for volume down, with internal pull-up resistors. The vibration motor is driven by PWM from BCM 13. Speech goes to the audio output and the bone-conduction headphone, and a USB microphone can be plugged in for voice commands. A power bank supplies 5 volts through the USB-C port. The UV lamp on the frame is switched by hand and is not wired to the Pi.");

