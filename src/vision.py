import os
import mss
import mss.tools

def capture_screen(output_path: str = "memory/screenshot.png") -> str:
    """Captures the primary monitor screen and saves it to output_path."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    with mss.mss() as sct:
        # monitor 0 is all monitors combined, monitor 1 is the primary monitor
        if len(sct.monitors) > 1:
            monitor = sct.monitors[1]
        else:
            monitor = sct.monitors[0]
            
        sct_img = sct.grab(monitor)
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_path)
        
    return output_path
