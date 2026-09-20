# Shortcuts

This page gives the key or the pointer action for each thing Gamut does. Most
keys are pressed alone, and the combinations use Ctrl.

## Where things are

- The Viewer, the large tab in the middle of the window, shows the picture.
- The Adjust tab is the panel on the right. Its sections, top to bottom, are
  Basic, Presence, Curve, Mixer, Grading, Masks, Presets and Versions, then the
  crop buttons.
- Along the bottom is the Timeline. Its keys and drags act on a video.
- The Mixer sets hue, saturation and luminance for each of eight colours.
- A preset is a saved look that any photo can take, from the Presets section.
- A version is a named saved state of one photo, kept in the Versions section,
  where Switch to brings one back.
- Pick is a button in the Masks section, on a luminance range or colour range
  mask.
- The Edit menu, beside File, holds Undo and Redo.

## Viewer: zoom

- The scroll wheel over the picture zooms in and out about the pointer. The
  spot under the pointer stays under it.
- One notch of the wheel changes the zoom by a factor of 1.25.
- Ctrl with the wheel does the same. A pinch on a touchpad zooms too.
- F fits the whole picture to the Viewer, and Ctrl+0 does the same.
- Ctrl+1 shows the picture at 100 percent, one screen pixel per picture pixel,
  about the middle of the Viewer.
- Ctrl+= zooms in one step about the middle of the Viewer, and Ctrl+- zooms out
  one step, by the same 1.25.
- The smallest zoom shows the whole picture, and at that size the picture is in
  Fit.
- A picture in Fit grows and shrinks with the window, while a zoomed picture
  keeps its percentage.
- Zooming all the way out puts the picture back in Fit.
- 800 percent is the largest zoom.
- From 400 percent up, single pixels are drawn as squares with no smoothing, so
  they can be judged.
- The top row of the Adjust tab shows the zoom as a percentage, with the buttons
  Fit, 100%, - and +, which do what the keys do for a person with no wheel.
- All of this works the same on a photo and a video frame.

## Viewer: pan

- Hold Space and drag with the left button to pan. The pointer shows a hand
  while Space is held.
- Drag with the middle button, the wheel pressed down, to pan without a key.
- A pan works wherever it starts, even on the crop or a mask handle, and moves
  neither.
- A picture in Fit has nowhere to go, so a pan does nothing until you zoom in.

## Crop and masks

- A plain drag inside the crop rectangle moves the crop.
- Dragging a handle of the selected mask moves that handle.
- A linear gradient has a start handle and an end handle. A radial gradient has
  a centre handle, one handle on each radius, and a rotation handle.
- Pick lights up when pressed and stays lit until your next click on the
  picture, which sets the range from the pixel under the pointer.
- The crop and the mask handles follow the pointer at every zoom.

## Tone curve and sliders

- A click on the tone curve line adds a point, and a drag moves one.
- Right-click a point to remove it. Delete removes it too, while the pointer is
  over it.
- The two end points cannot be removed.
- Click a slider, then Left and Right step it. The slider keeps the arrow keys
  from that click until you click somewhere else.
- With a video open, Left and Right move the slider and the playhead together.
  Click somewhere else to give the keys back to the Timeline.

## Undo and redo

- Ctrl+Z undoes.
- Ctrl+Shift+Z redoes, and Ctrl+Y redoes too.
- Undo and Redo in the Edit menu carry those shortcuts, and an entry is greyed
  out when there is nothing to undo or to redo.
- One drag is one step, however long it lasted, on a slider, a curve point, a
  colour wheel, a mask handle, the crop or a trim.
- A run of arrow-key presses on a slider becomes one step once the keys have
  rested for 0.4 seconds.
- Undo covers the sliders, the tone curve, the Mixer, the colour wheels, the
  crop, a preset applied, and masks added, changed, reordered or deleted. It
  covers anything done to a version, such as saving one or deleting one.
- On a video, a split, a trim and a deleted clip are covered as well.
- The history keeps up to 500 steps. It starts empty when a file opens, and it
  is gone when that file closes or another one opens.
- An undo leaves the zoom, the pan, the selected mask and the playhead alone.

## Timeline

- Space plays and pauses when you tap it, that is press and let go.
- A Space held while the picture is dragged is a pan, and letting it go does
  nothing.
- S splits the clip at the playhead.
- Delete removes the selected clip and closes the gap. It removes the tone curve
  point under the pointer as well, and it does both when both are there.
- To remove only the point, right-click it. To remove only the clip, keep the
  pointer off the curve.
- Home moves the playhead to the start and End moves it to the end. Both pause.
- Left and Right step the playhead one frame back and forward.
- A click or a drag on the timeline moves the playhead and pauses.
- Clicking a clip selects it, and dragging either end of a clip trims it.

## Saving and opening

- Drop a photo, a video or a .gamut project on the window to open it, or use
  File > Open.
- Gamut saves by itself, half a second after the last change and again when the
  window closes. There is no save key.
- A photo named example.heic gets a small edit file beside it,
  example.heic.gamut.json, and the photo itself is never changed.
- The project file for a video named clip.mp4 is clip.mp4.gamut beside it. It
  holds the cuts, the crop and the look, and the video itself is never changed.
- A .gamut project opens like a photo or a video.
- After an undo or a redo, the file beside the photo holds the picture as it
  then looks.
- The history itself is never written into either file.
- File > Export writes the finished picture as a JPEG, or the finished video as
  an MP4, to a place you choose.

## When a key is held back

- While you type in a text field, such as a preset or version name, the keys go
  to the text. F and Space are typed as characters, and the zoom keys do nothing
  to the picture.
- Ctrl+Z in a text field undoes your typing and leaves the picture alone.
- The Save, Discard, Cancel question appears when you press Switch to while the
  picture has changes that were not saved into any version.
- Gamut is then waiting to hear what to do with those changes, and the zoom
  keys, the pan and the undo keys wait for your answer.
