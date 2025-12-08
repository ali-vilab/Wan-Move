import os
import imageio.v3 as iio
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms

def draw_overall_gradient_polyline_on_image(image, line_width, points, start_color):
    """
    - image (Image): target image to draw on.
    - line_width (int): initial line width.
    - points (list of tuples): list of points forming the polyline, each point is (x, y).
    - start_color (tuple): starting color of the line (R, G, B).

    Return:
    - Image: original image with the gradient polyline drawn.
    """
    
    def get_distance(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    # Create a new image with the same size as the original
    new_image = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(new_image, 'RGBA')
    points = points[::-1]

    # Compute total length
    total_length = sum(get_distance(points[i], points[i+1]) for i in range(len(points)-1))

    # Accumulated length
    accumulated_length = 0

    # Draw the gradient polyline
    for start_point, end_point in zip(points[:-1], points[1:]):
        segment_length = get_distance(start_point, end_point)
        steps = int(segment_length)

        for i in range(steps):
            # Current accumulated length
            current_length = accumulated_length + (i / steps) * segment_length

            # Alpha from fully opaque to fully transparent
            alpha = int(255 * (1 - current_length / total_length))
            color = (*start_color, alpha)

            # Interpolated coordinates
            x = int(start_point[0] + (end_point[0] - start_point[0]) * i / steps)
            y = int(start_point[1] + (end_point[1] - start_point[1]) * i / steps)

            # Dynamic line width, decreasing from initial width to 1
            dynamic_line_width = int(line_width * (1 - (current_length / total_length)))
            dynamic_line_width = max(dynamic_line_width, 1)  # minimum width is 1 to avoid 0

            draw.line([(x, y), (x + 1, y)], fill=color, width=dynamic_line_width)

        accumulated_length += segment_length

    return new_image
   
def add_weighted(rgb, track):
    rgb = np.array(rgb) # [H, W, C] "RGB"
    track = np.array(track) # [H, W, C] "RGBA"
    
    # Compute weights from the alpha channel
    alpha = track[:, :, 3] / 255.0

    # Expand alpha to 3 channels to match RGB
    alpha = np.stack([alpha] * 3, axis=-1)

    # Blend the two images
    blend_img = track[:, :, :3] * alpha + rgb * (1 - alpha)
    
    return Image.fromarray(blend_img.astype(np.uint8))
        
def draw_tracks_on_video(video, tracks, visibility=None, track_frame=24):
    color_map = [
        (102, 153, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 102, 204),
        (0, 255, 0)
    ]
    circle_size = 12
    line_width = 16
    
    video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy() # (81, 480, 832, 3), uint8
    tracks = tracks[0].long().detach().cpu().numpy()
    if visibility is not None:
        visibility = visibility[0].detach().cpu().numpy()
    
    output_frames = []
    # Process the video
    for t in range(video.shape[0]):
        # Extract current frame
        frame = video[t]
        frame = Image.fromarray(frame).convert("RGB")
        
        # Draw tracks
        for n in range(tracks.shape[1]):
            if visibility is not None and visibility[t, n] == 0:
                continue
            
            # Track coordinate at current frame
            track_coord = tracks[t, n]
            tracks_coord = tracks[max(t-track_frame, 0):t+1, n]
            
            # Draw a circle
            draw = ImageDraw.Draw(frame)
            draw.ellipse((track_coord[0] - circle_size, track_coord[1] - circle_size, track_coord[0] + circle_size, track_coord[1] + circle_size), fill=color_map[n % len(color_map)])
            # Draw the polyline
            track_image = draw_overall_gradient_polyline_on_image(frame, line_width, tracks_coord, color_map[n % len(color_map)])
            frame = add_weighted(frame, track_image)
        
        # Save current frame
        output_frames.append(frame.convert("RGB"))
        
    return output_frames

def draw_mouse_track(track_video, tracks):
    mouse_icon_path = "assets/mouse_icon.png"  # replace with your icon path
    mouse_icon = Image.open(mouse_icon_path).convert("RGBA")
    icon_size = (64, 64)  # adjust icon size if needed
    icon_trans = (24, 16)
    mouse_icon = mouse_icon.resize(icon_size, Image.Resampling.LANCZOS)

    # Store processed frames
    output_frames = []

    for t in range(len(track_video)):
        # Convert to PIL image
        pil_frame = track_video[t].convert("RGBA")
        
        # Get the track coordinate at the current frame (assume using the first track)
        track_coord = tracks[0, t, 0].numpy()
        width, height = pil_frame.size  # note: size is (width, height)

        # Convert to pixel coordinates
        x = int(track_coord[0])
        y = int(track_coord[1])

        # Compute paste position (using the icon offset as reference)
        icon_w, icon_h = mouse_icon.size
        icon_trans_w, icon_trans_h = icon_trans
        paste_x = max(0, min(x-icon_trans_w, width - icon_trans_w))
        paste_y = max(0, min(y-icon_trans_h, height - icon_trans_h))

        # Paste the icon
        pil_frame.paste(mouse_icon, (paste_x, paste_y), mouse_icon)

        # Convert back to RGB and store
        final_frame = np.array(pil_frame.convert("RGB"))
        output_frames.append(final_frame)

    return output_frames

if __name__ == "__main__":
    save_dir = "saved_visuals"
    os.makedirs(save_dir, exist_ok=True)
    
    video_type = "image" # "video" or "image"
    fps = 16
    
    video_name = "Pexels_3C_product_0"
    video_path = f"MoveBench/en/video/{video_name}.mp4"
    track_path = f"MoveBench/en/track/single/{video_name}_tracks.npy"
    visibility_path = f"MoveBench/en/track/single/{video_name}_visibility.npy"
    
    frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"
    
    if video_type == "video":
        video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float() # [B, T, C, H, W]
    else:
        t = len(frames)
        video = torch.tensor(frames).permute(0, 3, 1, 2)[0:1].repeat(t, 1, 1, 1)[None].float()
    
    tracks = torch.tensor(np.load(track_path)).float()
    visibility = torch.tensor(np.load(visibility_path)).float()
    track_video = draw_tracks_on_video(video, tracks, visibility)

    track_video_with_mouse = draw_mouse_track(track_video, tracks)
    iio.imwrite(f"{save_dir}/{video_name}.mp4", track_video_with_mouse, fps=fps, plugin="FFMPEG")
