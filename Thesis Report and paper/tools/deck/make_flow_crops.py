# Image crops for the picture flowcharts on slides 3 and 4 (Problem Statement, Proposed Solution).
from pathlib import Path
from PIL import Image

A = Path(__file__).parent / "assets"
# the blind user with a cane, from the "today" street cartoon
Image.open(A / "cartoon_problem_3.jpg").crop((40, 20, 320, 334)).save(A / "flow_user.jpg", quality=92)
# the upper photo of the staged shop exchange with the v2 glass
im = Image.open(A / "v2_action_stack.jpg")
im.crop((0, 0, im.width, im.height // 2)).save(A / "flow_exchange.jpg", quality=92)
print("ok")
