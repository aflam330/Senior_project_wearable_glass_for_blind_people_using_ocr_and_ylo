$deck = "E:\Final SP\Thesis Report and paper\thesis report\slides\SARA_Presentation.pptx"
$out = "C:\Users\LAM\AppData\Local\Temp\claude\e--Final-SP\2865beee-89a6-4856-9b86-33aa83a25b55\scratchpad\deck\png"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($deck, $true, $false, $false)
$i = 1
foreach ($s in $p.Slides) { $s.Export((Join-Path $out ("s{0:D2}.png" -f $i)), "PNG", 1333, 750); $i++ }
$p.Close()
$app.Quit()
"exported $($i - 1)"
