# Shortcuts

This page lists Gamut's keys and pointer actions, group by group. Most keys are
pressed alone, and the combinations use Ctrl. The brush also uses Shift, with a
bracket key or a click, and Alt, held while painting.

## Where things are

- The Viewer, the large tab in the middle of the window, shows the picture.
- The Adjust tab is the panel on the right. Its sections, top to bottom, are
  Basic, Presence, Curve, Mixer, Grading, Masks, Presets and Versions, then the
  crop buttons.
- Along the bottom is the Timeline. Its keys and drags act on a video.
- The Mixer sets hue, saturation and luminance for each of eight colours.
- A preset is a saved look, kept in the Presets section. Apply puts it on the
  open picture.
- A version is a named saved state of one photo, kept in the Versions section.
  Each row there has a Switch to button, which brings that version back.
- Every mask is built from one or more components. A component is one shape or
  range that says where the mask applies. The kinds are linear gradient, radial
  gradient, luminance range, colour range and brush.
- The Masks section starts with five buttons: New linear, New radial, New
  luminance range, New colour range and New brush. Each one makes a mask with a
  single component of that kind and selects it.
- Click the name of a mask in the list at the top of the Masks section to select
  it. Its handles then show on the picture. A banner at the top of the Adjust
  tab reads "Editing mask:" with the name and a Done button.
- While a mask is selected, the sections Basic to Grading adjust that mask. Done
  goes back to the whole picture and leaves no mask selected.
- The components of the selected mask are listed under it. Each one has the
  buttons Add, Subtract and Intersect, which say how it combines with the
  components above it.
- At the bottom of the selected mask is the Add component row, with the buttons
  Linear, Radial, Luminance, Colour and Brush.
- Pick is a button on a luminance range or colour range component.
- The Edit menu, beside File, holds Undo and Redo.

## Viewer: zoom

- The scroll wheel over the picture zooms in and out about the pointer, on a
  photo and on a video frame alike. The spot under the pointer stays under it.
- One notch of the wheel changes the zoom by a factor of 1.25.
- Ctrl with the wheel does the same as the wheel alone.
- A pinch on a touchpad zooms too.
- F fits the whole picture to the Viewer. Ctrl+0 does the same.
- Ctrl+1 shows the picture at 100 percent, one screen pixel per picture pixel,
  about the middle of the Viewer.
- Ctrl+= zooms in one step about the middle of the Viewer. Ctrl+- zooms out one
  step. The step is the same 1.25.
- Fit is the smallest zoom, the one that shows the whole picture. Fit is also a
  state. A picture in Fit follows the window as it is resized, while a zoomed
  picture keeps its percentage.
- Zooming all the way out puts the picture back in Fit.
- 800 percent is the largest zoom.
- From 400 percent up, single pixels are drawn as squares with no smoothing, so
  they can be judged.
- The top row of the Adjust tab shows the zoom as a percentage, with the buttons
  Fit, 100%, - and +. They do what the keys and the wheel do.

## Viewer: pan

- Hold Space and drag with the left button to pan. The pointer shows a hand
  while Space is held.
- Drag with the middle button, the wheel pressed down, to pan without a key.
- A pan works wherever it starts, even on the crop or a mask handle. It moves
  neither of them. A pan works while you paint as well.
- A picture in Fit is seen whole. A pan does nothing until you zoom in.

## Crop and masks

- The crop rectangle is always on the picture. The crop buttons at the bottom of
  the Adjust tab are 4:5, 1:1, 3:4 and 9:16.
- Each crop button gives the largest rectangle of that shape that fits the
  picture, centred.
- A plain drag inside the crop rectangle moves the crop. With Paint on, that
  same drag paints and the crop stays where it is.
- The crop cannot be resized or drawn by hand. There is no free shape.
- Dragging a handle of the selected mask moves that handle.
- A linear gradient has a start handle and an end handle. A radial gradient has
  a centre handle, one handle on each radius, and a rotation handle.
- Pick lights up when pressed. It stays lit until your next click on the
  picture, which sets the range from the pixel under the pointer.
- The controls of the selected mask sit under the list of masks in the Masks
  section. The first is Opacity, a slider that sets how strongly the mask's
  adjustments are applied. It starts at 100, the top of its 0 to 100 range.
- Every mask has a slider named Refine edges, directly under Opacity. Its range
  is 0 to 100. A new mask has it at 0, which is off.
- Raise Refine edges and a loose edge of the mask moves onto the colour edge
  under it. A colour edge is a boundary inside the photo where one thing meets
  another, such as where a roof meets the sky. It is not the outer border of the
  photo.
- Paint a stroke loosely along a row of roofs, up into the sky. The mask lets go
  of the sky and stays on the roofs. A mask drawn a little short of a colour
  edge grows up to it.
- The slider sets how much of that move is taken. At 100 the mask takes the
  whole move. At 50 it goes halfway between the mask as drawn and the moved
  mask.
- A soft mask is not changed at all. With Refine edges on it looks the same as
  with Refine edges off. A wide gradient or a brush with a high Feather has a
  soft edge that fades out slowly. Refine edges acts on the mask's edge where it
  is fairly hard and lies near a colour edge.
- With no colour edge under the mask's edge, such as across open sky, the mask
  stays as drawn.
- While Refine edges is over 0, two sliders show under it, Radius and Edge
  sensitivity. At 0 they are hidden.
- Radius is how far the mask's edge may lie from a colour edge and still be
  moved onto it. It is shown as a percentage of the longer side of the picture.
  The lowest setting is 0.10 %, the highest is 5.00 %, and a new mask sits at
  1.00 %. A mask's edge further from a colour edge than the Radius stays put. A
  stroke that spills about 3 % of the longer side into the sky does not move at
  1.00 %. At 5.00 % the sky is let go.
- Edge sensitivity is how weak a colour edge can be and still hold the mask. It
  begins at 50, midway between 0 and 100. At 100 a faint colour edge, such as
  the rim of a thin cloud, holds the mask, and the mask's edge follows it. At 0
  only a strong colour edge, such as a roof against the sky, holds it.
- Refine edges acts on the whole mask, after its components are combined and
  after Invert. Subtract a brush from a gradient and the one shape the two make
  is refined. Any kind of component can be in that mask.
- It reads the picture as it was before any edit. Change Exposure or any other
  slider and the refined mask holds the edge it already has.
- Under Opacity, top to bottom, the controls of every mask are Refine edges,
  then Radius and Edge sensitivity while Refine edges is over 0. Shift edge,
  Feather and Contrast follow, in that order. These three are always shown,
  whatever Refine edges is set to. Invert and the Show overlay checkbox sit side
  by side below them.
- Shift edge moves the edge of the mask outward or inward. Set it over 0 and the
  mask gets larger. Set it under 0 and the mask gets smaller.
- The value of Shift edge is how far the edge moves, shown as a percentage of
  the longer side of the picture. It can be set anywhere between -5.00 % and
  5.00 %. On a new mask it sits at 0, and at 0 the edge stays where it is.
- Shift a soft edge and it stays soft. The fade moves with the edge and keeps
  its width. Shift a hard edge and it stays hard.
- A stroke along a row of roofs can spill into the sky in a thin strip that
  Refine edges does not take back. Pull Shift edge under 0 by about the width of
  the strip, and the strip goes. The rest of the edge of the mask moves in by
  the same amount.
- Feather is a slider of the mask, separate from the Feather setting under the
  brush component.
- The Feather slider of the mask softens every edge of the mask. That includes a
  hard edge left by Refine edges. Its value is the width of the fade, as a
  percentage of the longer side of the picture.
- The Feather slider of the mask goes up to 5.00 %. Feather is off at 0, its
  lowest setting and the one a new mask starts with.
- Contrast makes a soft edge firmer. Raise it and the fade narrows about the
  middle of the edge. At 100 the edge is hard.
- Contrast is set on a scale of 0 to 100. A new mask starts with Contrast at 0,
  and 0 changes nothing.
- The edge is shifted first, then feathered, then its contrast is raised.
  Feather a hard edge and then raise Contrast, and the fade comes back toward
  hard at the setting you choose.
- Shift edge, Feather and Contrast act on the whole mask, after its components
  are combined, after Invert and after Refine edges. They act on any kind of
  mask, whatever components it holds.
- Moving Exposure or any other slider of the Adjust tab leaves a shifted,
  feathered or contrasted edge where it is. The three work from the shape of the
  mask alone. With all three at 0, the mask is exactly as drawn, or as Refine
  edges left it.
- Shift edge and Feather move in small steps near 0 and in larger steps toward
  their ends. That makes a fine setting easy to pick. The smallest setting off 0
  is 0.01 %.
- Shift edge and Feather are a share of the picture's longer side. The edge of
  the mask therefore looks the same in the fitted view, zoomed in, and in the
  exported file.
- The overlay is a red tint where the selected mask applies. The Show overlay
  checkbox, beside Invert under Contrast, turns it on and off. With Refine edges
  on, the overlay shows the mask as refined. With Shift edge, Feather or
  Contrast set off 0, it shows the mask as shifted, feathered and contrasted, in
  place of the mask as drawn. Turn it on to watch the edge move as you drag any
  of these sliders.
- Refine edges, Radius, Edge sensitivity, Shift edge, Feather and Contrast are
  saved with the mask in the picture's file. One drag of any of the six is one
  undo step. None of the six has a key.
- Paint into a brush mask with Refine edges on and the new paint is refined as
  you paint. With Shift edge, Feather or Contrast set off 0, the new paint is
  shifted, feathered and contrasted as you paint too.
- On a video the mask is refined again on every frame. Its edge follows the
  colour edges of each frame.
- Shift edge, Feather and Contrast are applied again on every frame of a video,
  after Refine edges.
- Zoomed in or out, the crop and a mask handle move exactly as far as the
  pointer moves.

## Brush

- The Paint button of a brush component puts the brush in your hand. The button
  then reads "Painting...".
- Esc puts the brush down. So does a second press of Paint, the Done button of
  the mask, selecting another mask, and the Save, Discard, Cancel question that
  Gamut asks when you switch versions with unsaved changes.
- Pressing Brush in the Add component row adds a brush component to the mask you
  are editing.
- Subtracting a brush from a radial gradient takes the painted part out of the
  gradient.
- The settings sit under the brush component in the Masks section.
- Size is the radius of the brush, as a percentage of the longer side of the
  picture. It runs from 0.02 to 50.
- Feather runs from 0 to 100. It is how much of the radius the edge fades over.
  0 is a hard edge, and 100 fades from the centre.
- Flow runs from 1 to 100. It is how much one pass paints. Passes over the same
  place build up.
- Under Flow there is a checkbox named Auto mask. It is unticked when Gamut
  starts.
- The brush paints a stroke as a row of round marks laid close together, called
  dabs. With Auto mask ticked, each dab paints only what is like the colour
  under the centre of the brush at that dab.
- The ring is the round pointer that shows the size of the brush over the
  picture. Paint along the sky beside a tower and the stroke stops at the
  tower's edge, even where the ring overlaps the tower.
- The colour is read where the centre of the ring is. Keep the centre on the
  thing you want painted. Move the centre onto the tower and the dabs there
  paint the tower.
- A stroke across an area whose colour changes slowly paints all of it, because
  each dab takes the colour under its own centre. Everything the ring covers
  there is close to the colour under its centre, so every dab paints all it
  covers. Auto mask changes nothing you can see on such an area. The difference
  appears where the brush crosses a clear edge, such as a roof against the sky.
- While Auto mask is ticked, a Sensitivity slider shows under it. It runs from 0
  to 100 and starts at 50.
- A higher Sensitivity keeps the stroke to colours closer to the one under the
  centre. At 100 a stroke over blue sky leaves thin white cloud unpainted. At 0
  it takes in nearly everything the ring covers.
- The colour compared is the colour of the picture before any edit. Moving a
  slider never moves what an auto stroke covers.
- Under Auto mask there is a row that reads "Pressure:" with two checkboxes,
  Size and Flow. Flow is ticked and Size is unticked when Gamut starts.
- The two Pressure checkboxes act with a pen that reports pressure.
- With the Pressure checkbox Flow ticked, a lighter hand paints less and a
  heavier hand paints more, up to the Flow setting. The lightest touch still
  paints a twentieth of the Flow setting, so a light hand always leaves a mark.
- With the Pressure checkbox Size ticked, a lighter hand paints a smaller dab.
  The dab runs from a fifth of the Size setting at the lightest touch to the
  full Size at full pressure.
- Both Pressure checkboxes can be ticked together.
- The ring shows the full Size, not the size at the current pressure.
- A stroke keeps the size, feather, flow, Auto mask setting and Sensitivity it
  was painted with. Change a setting afterwards and the change reaches the next
  stroke only.
- The settings belong to the brush tool, Auto mask, Sensitivity and the two
  Pressure checkboxes included. They are not saved with the picture, undo does
  not change them, and they are back at their starting values when Gamut starts
  again.
- A plain drag with the left button paints one stroke, anywhere on the picture.
  That includes inside the crop rectangle.
- While you paint, the handles of the mask's other components are hidden.
- A click without a move paints one dab.
- Shift with a click paints a straight line from the end of the last stroke to
  the click. With nothing painted yet, it paints one dab.
- A mouse and a finger on a touch screen paint at full pressure, whatever the
  two Pressure checkboxes say.
- One touch paints. A second finger on the screen ends the stroke where it is,
  because two fingers are a pinch and a pinch zooms.
- A stroke painted with a pen keeps the pressure of every point in the saved
  file. It keeps which of the two Pressure checkboxes were ticked when it began.
- Alt held during a drag makes that stroke erase. The Erase checkbox beside
  Paint does the same for every stroke until it is unticked.
- An erase stroke takes away what was painted in that brush before it. It is an
  ordinary stroke, not an undo. Paint over the same place afterwards and there
  is paint there again.
- Auto mask works with Erase and with Alt. An auto erase stroke takes paint away
  only from what is like the colour under the centre.
- Both pans and the wheel work as always while the brush is up. The pointer is
  the hand while Space is held. Painting works at every zoom.
- [ makes the brush smaller and ] makes it larger. Each press is a factor of
  1.1.
- Shift+[ lowers the feather by 5. Shift+] raises it by 5.
- O shows and hides the overlay. The Show overlay checkbox does the same.
- A is the key for Auto mask while the brush is in your hand. A alone ticks and
  unticks the checkbox. With Shift or Ctrl held, A does not tick the checkbox.
  Ctrl+A keeps its usual job in a text field, where it selects the text.
- The overlay is red where the mask applies. Pressing Paint turns it on, so you
  see what you paint.
- Putting the brush down returns the overlay to what it was before.
- The pointer over the picture is a ring the size of the brush at the current
  zoom. It shows what a dab will cover.
- An inner ring marks where the feather starts.
- A short dash in the middle of the ring means the next stroke erases.
- With Auto mask on, the ring has a small cross at its centre. The cross marks
  where the colour is read.
- A brush too small to draw as a ring shows as a small cross in place of the
  ring.
- The section shows how many strokes the brush holds, beside a Clear strokes
  button, which removes them all.
- One brush holds up to 2,000 strokes. At 2,000 Gamut shows a message and the
  brush takes no more. To go on, add another brush component from the Add
  component row and paint in that one.
- A stroke is kept as the path the pointer took rather than as pixels, so
  strokes stay sharp at every zoom and in the export.
- On a video the brush paints the frame under the playhead. The strokes stay
  where they were painted and do not follow motion.
- With Auto mask, every frame is checked again inside those strokes. The part of
  each stroke that is covered can change from frame to frame as colours move
  under it. The strokes themselves do not move.

## Tone curve and sliders

- A click on the tone curve line adds a point, and a drag moves one.
- Right-click a point to remove it. Delete removes it too, while the pointer is
  over it.
- The two end points cannot be removed.
- Click a slider, then Left and Right step it. The slider keeps the arrow keys
  until you click somewhere that is not a slider, an empty part of the Adjust
  tab for example.
- With a video open, one press of Left or Right then moves the slider one step
  and the playhead one frame, both at once. To move only the playhead, click off
  the slider first. To move only the slider, drag it with the mouse.

## Undo and redo

- Ctrl+Z undoes.
- Ctrl+Shift+Z redoes, and Ctrl+Y redoes too.
- Undo and Redo in the Edit menu carry those shortcuts. An entry is greyed out
  when there is nothing to undo or to redo.
- One drag is one step, however long it lasted, on a slider, a curve point, a
  colour wheel, a mask handle, the crop or a trim.
- A brush stroke is one step too.
- Clear strokes is a step. So is deleting a brush component, and undoing that
  brings the strokes back.
- An undo that takes away the brush component puts the brush down.
- While you keep pressing the arrow keys on a slider, no step is made. Once 0.4
  seconds pass with no press, all the presses of that run become one step. One
  Ctrl+Z takes the whole run back.
- Undo covers the sliders, the tone curve, the Mixer, the colour wheels, the
  crop, a preset applied, and masks added, changed, reordered or deleted.
- Saving, updating, renaming, deleting and switching to a version can each be
  undone.
- On a video, a split, a trim and a deleted clip are covered as well.
- The history keeps up to 500 steps. It starts empty when a file opens, and it
  is gone when that file closes or another one opens.
- An undo leaves the zoom, the pan, the selected mask and the playhead alone.

## Timeline

- Space plays and pauses. Press it and let it go with no drag in between, and it
  acts however long it was held.
- A Space held while the picture is dragged is a pan. Letting it go then does
  nothing.
- On a photo, Space alone does nothing.
- S splits the clip at the playhead.
- Delete removes the selected clip and closes the gap. Delete also removes the
  tone curve point under the pointer, so with a clip selected and the pointer
  over a curve point one press removes both.
- To remove only the point, right-click it. To remove only the clip, keep the
  pointer off the curve.
- Home moves the playhead to the start and End moves it to the end. Both pause.
- Left and Right step the playhead one frame back and forward. A slider you
  clicked takes those keys as well, under the slider rule above.
- A click or a drag on the timeline moves the playhead and pauses.
- Clicking a clip selects it, and dragging either end of a clip trims it.

## Saving and opening

- Drop a photo, a video or a .gamut project on the window to open it. File >
  Open does the same.
- File > Open and File > Export are menu entries. Neither has a key.
- Gamut saves by itself. It writes the edit once half a second has passed with
  no new change, and again when the window closes. There is no save key.
- A crash or a forced quit loses only a change made in that half second, or a
  drag that was still under way.
- A photo named example.heic gets a small edit file beside it,
  example.heic.gamut.json. The photo itself is never changed.
- That file is not a project and is not opened by hand. Open the photo, and
  Gamut reads it.
- The project file for a video named clip.mp4 is clip.mp4.gamut beside it, with
  no .json on the end. It holds the cuts, the crop and the look, and the video
  itself is never changed.
- A .gamut project is a file you can open or drop.
- Brush strokes are saved in those files like every other edit. Close and open
  the file, and the strokes are there.
- After an undo or a redo, the file beside the photo holds the picture as it
  then looks.
- The history itself is never written into either file.
- File > Export writes the finished picture as a JPEG, or the finished video as
  an MP4, to a place you choose.

## When a key is held back

- While you type in a text field, such as a preset or version name, the keys go
  to the text. F and Space are typed as characters, and the zoom keys do nothing
  to the picture.
- The brush keys [, ], O, A and Esc go to the text as well, and leave the brush
  alone.
- Ctrl+Z in a text field undoes your typing and leaves the picture alone.
- With Paint off, [, ], O and A do nothing.
- Ctrl with [, ], O or A is not a brush key. It does nothing to the brush.
- The Save, Discard, Cancel question appears when you press Switch to in the
  Versions section while the picture has changes that were not saved into any
  version.
- While that question is on the screen, the zoom keys, the pan, the undo keys
  and the brush do nothing. Answer it and they work again.
