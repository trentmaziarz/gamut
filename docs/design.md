# Gamut: design document

Version 1, 2026-09-15. Decisions D-1 to D-8 are ruled. Everything else is the author's plan.

Gamut is a free, open source editor for phone content that covers what people actually use in Lightroom and DaVinci Resolve when they post to Instagram. It is a native Windows application. The user it serves shoots on a phone and wants a 4:5 still and a 9:16 reel without a subscription. Photo develop and the video timeline live in one app over one engine.

The engine is the six crates below gamut-app, named core, color, gpu, media, ai and templates. It has no window and no widgets. It turns a file and a set of edit parameters into pixels. The rulings below date from 2026-09-15, and the rest of this plan rests on them.

1. (D-1) Gamut is a public product for other people, free and open source under a dual MIT or Apache-2.0 license.
2. (D-2) The name is Gamut, the workspace is C:/gamut, and the repo is trentmaziarz/gamut on GitHub, public.
3. (D-3) Photo develop and the video timeline live in one app over one engine from the first commit.
4. (D-4) The language is Rust with wgpu, and the interface layer is egui through the egui-wgpu integration.
5. (D-5) Video decode and encode run through the ffmpeg libraries, linked dynamically, with NVDEC and NVENC hardware acceleration.
6. (D-6) Version 0.1 opens phone photos and video, phone HDR video tone-mapped for SDR export, camera RAW, and existing edits.
7. (D-7) Setup is one download and one double-click.
8. (D-8) Pre-made SVG templates for Buhurt posts and recipe posts ship with the app and are the test corpus.

The author has no existing workflow to copy, so the feature set comes from what Instagram creators use in Lightroom, Resolve and CapCut. Milestones alternate between the photo side and the video side. Every milestone ends with a tagged release that builds and passes its tests on CI, so the app is usable at every tag.

The toolkit choice under D-4 came with one constraint, given in the words "speed of build isn't the problem, it's an optimized product". egui and the engine share one wgpu device. The rendered frame is drawn into the window with no copy across a process or a webview boundary. egui redraws the whole interface every frame, and that is the usual doubt about it for a large application. Rerun answers the doubt. Rerun is an open source viewer for robotics and computer vision data streams, built by the team that wrote egui. It draws large 2D, 3D and time-series views on egui over wgpu at interactive frame rates. That is a heavier load than a photo and a timeline. egui_dock and the painter API supply docking, curve editors and scopes that iced would need built from scratch.

Version 0.1 opens phone photos and video in JPEG, HEIC, H.264 and HEVC. It opens phone HDR video in Dolby Vision and HLG, tone-mapped for SDR export. It opens camera RAW in CR3, ARW and DNG, and edits already made in MP4, MOV and PNG. Those existing edits are files another tool has already exported. Gamut ships as a portable zip and as a per-user installer that needs no admin rights.

The shipped templates cover Buhurt posts and recipe posts. Buhurt is armored medieval combat. One recipe template scrolls a quote from the original source.

Lightroom is the most used photo tool among influencers, by a HypeAuditor survey of 1,200 influencers on photo editing tools. The free edition of DaVinci Resolve caps at 8-bit formats, UHD 3840x2160 at 60 fps and one GPU. GPU H.264 and H.265 encode, Magic Mask, noise reduction, voice isolation, Smart Reframe and "Create Subtitles from Audio" are Studio only. CapCut's paid plan cost $77.99 a year until late 2025. By February 2026 the plan with the same features cost $179.99 a year, 2.31 times the old price. Creators who left CapCut over that price are the first users Gamut is built for. The CapCut free tier now gates auto-captions and background removal behind paid credits, watermarks Pro templates and has no 4K.

No open source editor reproduces the CapCut bundle of templates, captions and effects. Darktable and RawTherapee are module-heavy with no preset library. Kdenlive and Shotcut have weak text animation and no social presets. Olive has no usable build. Blender VSE cannot animate the content of a text strip, and LosslessCut is cut-only. Instagram says 85 percent of Reels are watched muted. That statistic is why auto captions sit first in the feature order. Captions need the timeline from M4 and gamut-ai from M6 first, which is why they ship in M7.

Commits are small, frequent and real, with human-style messages and no backdating. Version 0.1 is the first public announcement, and the tags before it are working releases.

## What ships

The order of the 25 features below is the author's, set in September 2026. It comes from the vendor feature pages, the HypeAuditor survey of photo editing tools and the platform statistics cited above. No usage measurement exists behind it, and a reader may dispute the order item by item.

1. Auto captions burned into the video.
2. A 9:16 1080x1920 canvas with the interface safe zones drawn on it.
3. Trim, split and multi-track cutting.
4. Photo presets and saved looks.
5. Basic tone sliders for exposure, contrast, highlights and shadows.
6. Music sync and trending audio.
7. Text overlays with animation presets.
8. Transitions between clips.
9. Speed ramps on preset curves.
10. The 4:5 and 3:4 photo crops for the 2025 grid change.
11. The HSL color mixer.
12. LUTs and filters applied to video.
13. Export presets, H.264 at 30 fps for Reels and sRGB JPEG for posts.
14. AI subject, background and sky masks.
15. Object and blemish removal.
16. Background removal and cutout in video.
17. Music ducking under a voice track.
18. Stickers, overlays and emoji.
19. Templates for trend edits.
20. Auto reframe of 16:9 footage to 9:16.
21. Keyframed position and scale zooms.
22. Color wheels and curves grading.
23. Radial and linear gradient vignettes.
24. Lens and background blur for portraits.
25. Voice and noise cleanup.

Version 0.1 is nine milestones, M0 to M8, and it is done when all nine are done. Each one ends with a tagged release and something a user can do. The bracketed numbers point back to the list above. The week counts are the author's estimates of evenings and weekends on the reference machine. They total 32 weeks. They are calendar weeks of evenings-and-weekends work. Counted from 2026-09-15 that lands on 2027-04-27, which puts version 0.1 in the second quarter of 2027.

M0 Scaffold. A Cargo workspace, CI on GitHub Actions running windows-latest, the license files and a README. One egui window draws a test image on a wgpu canvas. Tag v0.0.1. The build runs on CI and the window opens. Estimated at 1 week.

M1 Photo, first light. Opens JPEG, PNG and HEIC. The Basic panel runs on the GPU with white balance, exposure, contrast, highlights, shadows, whites, blacks, vibrance and saturation. Crop to 4:5, 1:1, 3:4 and 9:16, under a phone-frame preview with the Instagram safe zone drawn on it. Export JPEG in sRGB at 1080x1350, 1080x1080, 1080x1440 and 1080x1920, quality 85. A user can develop a phone photo and post it. Creator features: tone sliders [5], grid crops [10], photo export [13]. Estimated at 3 weeks.

M2 Video, first cut. Opens H.264 and HEVC in MP4 and MOV through ffmpeg with NVDEC. One video track with trim, split and ripple delete. Scrub and play with audio over the same phone-frame preview. Export a Reel through NVENC as H.264 High at 1080x1920, at 30 fps or the source rate. Audio is AAC at 128 kbps and 48 kHz, and the file carries the moov atom first, a closed GOP and 4:2:0. A user can cut a phone clip and upload the reel. Creator features: the 9:16 canvas [2], cutting [3], video export [13]. Estimated at 4 weeks.

M3 Photo, the look. The tone curve is a point curve per channel with monotone cubic interpolation. The HSL mixer covers eight hue ranges. The grading wheels cover shadows, midtones and highlights on a lift, gamma and gain model. Texture, clarity and dehaze land here too. Masks come by brush, linear gradient, radial gradient, luminance range and color range, and they combine by add, subtract and intersect. The brush shows a red overlay while painting. Presets save and apply as JSON, and versions keep named alternates of one photo. Creator features: presets [4], HSL [11], wheels and curves [22], gradient vignettes [23]. Estimated at 4 weeks.

M4 Video, the timeline. Multiple video and audio tracks. Transitions for cut, dissolve, dip to black and wipe. Keyframed position, scale, rotation and opacity. Speed ramps on preset curves. Waveform display, audio gain and fades, and music ducking under a voice track. A .cube LUT per clip, and the M3 grade panel per clip. Creator features: music sync [6], transitions [8], speed ramps [9], LUTs [12], ducking [17], keyframed zooms [21]. Estimated at 5 weeks.

M5 Templates and overlays. SVG overlays rendered by resvg, and text layers rendered by glyphon with the system fonts. Text animation presets for in, out and combo motion, covering fade, slide, pop and typewriter. Stickers and the recipe-quote scroll template ship here. The Buhurt pack holds a lower-third name plate, a score card, an event title and a bracket slide. Creator features: animated text [7], stickers [18], templates [19]. Estimated at 3 weeks.

M6 Photo, the camera. Camera RAW through rawler for CR3, ARW and DNG, with demosaic, camera matrix and white balance. HDR gain-map JPEG and HEIC read on import. Lens correction for distortion, vignetting and chromatic aberration from a bundled profile table. Transform by rotate and perspective, effects for vignette and grain, and a remove tool with heal and clone. AI subject, background and sky masks through ONNX Runtime with DirectML, plus lens blur driven by the subject mask. Creator features: AI masks [14], removal [15], lens blur [24]. Estimated at 5 weeks.

M7 Video, the phone. Dolby Vision 8.4, HLG and PQ sources tone-mapped with the BT.2390 EETF on the GPU. Auto reframe of a 16:9 source into 9:16 by subject mask. Auto captions through whisper.cpp with styled caption templates. Video background removal from the subject mask, and voice noise cleanup by spectral gate. Creator features: captions [1], video cutout [16], auto reframe [20], voice cleanup [25]. Estimated at 5 weeks.

M8 Packaging and release. The Velopack per-user installer and the portable zip, each built by a release workflow on tag. A SmartScreen note in the README, an application to SignPath Foundation for free code signing, tag v0.1.0 and announce. Estimated at 2 weeks.

Instagram Reels and Stories take 1080x1920 at 9:16, in MP4 or MOV with the moov atom first and no edit lists. The codec is H.264 or HEVC, progressive, closed GOP, 4:2:0, VBR up to 25 Mbps, 23 to 60 fps. Audio is AAC up to 48 kHz at 128 kbps in 1 or 2 channels. Reels run up to 300 MB by the API, Stories up to 100 MB and 3 to 60 seconds. Feed images are 1080x1080 (1:1), 1080x1350 (4:5) or 1080x566 (1.91:1), JPEG, 8 MB, width 320 to 1440. The profile grid has cropped previews to 1080x1440 at 3:4 since January 2025. 4:5 stays the recommended feed post, and carousels hold up to 20 items. TikTok takes 1080x1920 and recommends H.264, with a third-party safe zone of about 130 px top, 484 px bottom and 100 px right. YouTube Shorts takes 1080x1920 up to 3 minutes, H.264 High at 8 Mbps for 1080p.

Meta's Instagram Reels and Stories ads guides give the safe zone for 1080x1920. They keep the top 14 percent, the bottom 35 percent and 6 percent each side clear. That is about 269 px top, 672 px bottom and 65 px each side. Gamut draws this on the phone-frame preview and can toggle a 3:4 grid-crop guide on a 4:5 photo.

iPhone 15 to 17 default to High Efficiency, so photos are HEIF and video is HEVC. HDR Video is on by default, and Dolby Vision requires HEVC. The file is Dolby Vision profile 8.4 with an HLG base layer, 10-bit HEVC, BT.2020, in a .mov with colr and dvvC boxes. Pro models add ProRes, ProRes RAW, Apple Log and 48 MP ProRAW DNG. Samsung S24 and S25 default to HEVC, and HEIF photos are a toggle with JPEG otherwise. They record HDR10+ video in PQ and save Super HDR photos as gain-map JPEGs. Orientation lives in both the HEIC container transforms and EXIF. MOV rotation is a track matrix that ffmpeg applies on decode. Live Photos pair a still and a MOV by a content identifier. Gamut reads all of these on import.

HDR to SDR has no single agreed method. ffmpeg offers a CPU chain that converts with zscale to linear, tone-maps with hable, mobius or reinhard, then retags to BT.709. It also offers a GPU chain through libplacebo with the BT.2390 EETF, soft-clip gamut mapping and peak detection at the 99.995 percentile. Resolve tone-maps only when color management is on, and it assumes video range while iPhone files are full range. CapCut publishes no method. Gamut implements BT.2390 in its own WGSL shader in gamut-gpu, with the CPU reference in gamut-color.

oximedia-hdr was created in March 2026 and had 276 downloads on 2026-09-15. No other published crate depends on it, so Gamut would be its first serious user and would be debugging it alongside Gamut itself. colstodian last released in 2021 and kolor in 2023, so both are dead. moxcms already supplies the PQ and HLG transfer functions, which leaves only the tone-mapping curve to write. Instagram accepts HLG and PQ uploads and derives SDR itself. iPhone Dolby Vision 8.4 uploads have been transcoded to AV1 with Dolby Vision 10.4 metadata since June 2025. Version 0.1 exports SDR only, and HDR export is a version 0.2 item.

## How it is built

The code is a Cargo workspace of seven crates under crates/.

gamut-core holds the project document. It carries the edit parameters for a photo and the timeline model for a video. The timeline model covers tracks, clips, in and out points, speed curves, keyframes, transitions and overlays. Presets and undo history live here too. It is plain data with serde, no GPU and no I/O. A photo edit is stored as a sidecar JSON next to the original, in the way Lightroom keeps XMP. A video project is a .gamut JSON file with media paths relative to it. Presets are JSON subsets of the photo parameters. All three are text, so they diff in git.

gamut-color is the color science on the CPU. It holds matrices between sRGB, Rec.709, Rec.2020, ProPhoto and XYZ, and transfer functions for sRGB, Rec.709, PQ and HLG. It holds Bradford chromatic adaptation for white balance, the BT.2390 tone-mapping curve, .cube LUT parsing and ICC profiles through moxcms. It also holds ACEScct. ACEScct is a log encoding of linear light published by the Academy of Motion Picture Arts and Sciences as part of ACES. It runs as a straight line below 0.0078125 and a log curve above, and its constants are public. Gamut works in it because curve and wheel controls then move shadows and highlights by similar visual amounts. Every operator in this crate is the reference its GPU shader is tested against.

gamut-gpu is the wgpu render graph and the WGSL shaders. Textures are rgba16float in the working space and masks are r8unorm. The video compositor lives here too, with transforms, transitions, overlays, text through glyphon, SVG rasterized by resvg into textures, and readback for export.

The render graph draws into an offscreen texture that egui then draws as an image. egui gives a crate one slot inside its own drawing pass. That slot has one output image and no scratch images of its own. Gamut's renderer runs several passes and needs scratch images between them. It draws into its own texture first and hands egui the finished picture. The cost is one extra draw of that picture per frame. The picture never leaves the GPU.

gamut-media is decode and encode. ffmpeg-the-third covers containers, H.264, HEVC and audio. One unsafe module handles the hardware device context, because no safe ffmpeg wrapper exposes hwaccel. libheif-rs covers HEIC and AVIF, rawler covers RAW, and the image crate covers JPEG and PNG. Audio decodes through ffmpeg rather than symphonia, which has no Opus or HE-AAC. cpal plays back and rubato resamples. Waveform peaks come from min and max binning written in the crate, because no maintained peak crate exists. A frame cache runs per source, and optional 1080p proxies are generated on import for 4K sources.

gamut-ai is ONNX Runtime through the ort crate with the DirectML execution provider. DirectML needs only a DX12 GPU and ships as two DLLs beside the exe, with no admin install. A U2-Net or MobileSAM model gives subject masks and whisper.cpp gives captions, each downloaded on first use. The crate sits behind a feature flag, so the app builds without it.

gamut-templates owns the template format. A template is a folder holding template.svg and template.toml. The TOML names the text slots, the fonts, the duration and the keyframes as property, time, value and easing. resvg renders static SVG and animates nothing, so all motion lives in the TOML sidecar. Text slots render through glyphon at runtime and are never baked into the SVG, so wrapping and fonts work. The recipe-quote scroll is a keyframed vertical translation whose duration comes from the word count at 200 words per minute.

gamut-app is the egui application. One window, egui_dock panels, a Photo view and a Video view over the same engine, the phone-frame preview, the export dialog. Startup target is under 2 seconds.

The pipeline is a fixed order of operations with masks, the Lightroom model rather than the Resolve node graph. Creators work from presets, and a preset is portable only when the order is fixed. Nodes are a Resolve power feature that the top 25 list never mentions.

One frame runs through these steps in this order.

- Decode, then the input transform into the working space.
- Lens correction and geometry.
- White balance by Bradford chromatic adaptation.
- Exposure as a multiply by 2 to the power EV.
- Highlights, shadows, whites and blacks.
- Contrast as an S-curve around 18 percent gray.
- Texture and clarity as local contrast at two scales.
- Dehaze by the dark channel prior.
- Tone curve, HSL and color wheels in the ACEScct log domain, the wheels on the lift, gamma and gain form of the ASC CDL.
- Vibrance and saturation.
- Detail, meaning sharpen and denoise.
- Effects for vignette and grain.
- Each mask re-runs its own subset of the parameters above and blends by mask alpha.
- The output transform to sRGB or Rec.709, with tone mapping when the source is HDR.
- Dither, then encode.

The highlights and shadows step works on a two-layer split. The base layer is a blurred copy holding the large-scale brightness, and the detail layer is the image minus the base. The highlights and shadows sliders compress the base layer in its bright or dark range. The detail layer is added back unchanged, so edges and texture survive. Whites and blacks set the clipping endpoints.

For video the same per-clip pipeline feeds the compositor. The compositor applies the keyframed transform, the transition, the overlays and the text, then the export encoder runs.

The working space is linear Rec.2020 at a D65 white point in 16-bit float. Phone HDR video is already encoded in BT.2020, so it needs no gamut conversion on input. Phone SDR video and photos are sRGB or Display P3, and each of those sits inside Rec.2020. A RAW file has no color space of its own. Every RAW converter multiplies by the camera matrix to reach XYZ, then by one more matrix to reach the working space. Choosing Rec.2020 adds nothing to that step. Camera sensors can record colors outside Rec.2020, and those arrive as negative or above-one values. The 16-bit float format keeps them as they are, and nothing is clipped until the output transform. Lightroom uses linear ProPhoto at D50, and Resolve uses DaVinci Wide Gamut with the DaVinci Intermediate log curve. Both of those log curves are private to their vendor, while ACEScct is published with its constants. Gamut therefore runs its curves and wheels in ACEScct.

The playback path in version 0.1 accepts one copy. NVDEC decodes to GPU memory, the frame is copied to system memory, and the result is uploaded to wgpu. Zero-copy sharing of the decoded texture is a version 0.2 item. The copy is expected to be fast enough for 4K30 on the reference machine. That expectation is an estimate. Timing this copy is the first task of M2. If the copy misses the 4K30 target, zero-copy sharing moves from version 0.2 into M2.

The crate versions were checked against crates.io on 2026-09-15.

- ffmpeg-the-third 6.0.0 tracking FFmpeg 9.0, released 2026-08-04, WTFPL, an active fork, while ffmpeg-next is maintenance-only.
- The ffmpeg DLLs are found through FFMPEG_DIR pointing at a BtbN or gyan.dev shared LGPL build, and bindgen needs LIBCLANG_PATH.
- egui-wgpu and eframe 0.36.2 on wgpu 30, egui_dock 0.21.1, egui_plot 0.37.0.
- rawler 0.8.0, LGPL-2.1, about 47 Canon CR3 bodies and 70 plus Sony ARW bodies, demosaic included.
- Canon R5 II and Sony a1 II are not confirmed in rawler's list.
- libheif-rs 3.0.0 wrapping libheif 1.23.1, built through vcpkg with libde265 and dav1d.
- The only capable pure-Rust HEIC decoder is AGPL and is excluded.
- resvg 0.48.1, static SVG only, filters supported, no animation.
- cpal 0.18.2 and rubato 5.0.0 for audio, moxcms 0.9.1 for ICC and the PQ and HLG transfer functions.
- glyphon 0.12.0 on wgpu 30 for text, with vello excluded because it lags one wgpu major behind egui-wgpu.
- ort 2.0.0-rc.13 with ONNX Runtime 1.28 on DirectML, which allows one run per session at a time and caps at opset 20.
- Velopack 1.2.110 installs per-user under LocalAppData with no elevation and gives delta updates. Its vpk tool needs the .NET 8 SDK, which is installed in M8, since the build machine has only the .NET 6 runtime today.

rawler returns an error on a body it does not know. Gamut then shows the camera model read from the file's metadata. The message says the body is not supported yet and links to the dnglab supported-cameras page. The user can convert the file with Adobe's free DNG Converter, and rawler reads the DNG that comes out.

ffmpeg and libheif are LGPL, so they are linked dynamically and their DLLs ship in the zip. rawler is LGPL-2.1 and is a Rust crate, so it links statically. LGPL 2.1 section 6 lets a program link the library statically. The condition is that the user can modify the library and relink the program with the modified version. Gamut ships its complete source under MIT or Apache-2.0, so a user can change rawler, run cargo build and get a working Gamut. That rebuild is the relink the license asks for. The plan proceeds on this reading. If a legal review before version 0.1 disagrees, rawler is built as a separate DLL that Gamut loads at run time. The LGPL allows that without question, and it costs about a week. whisper.cpp and ONNX Runtime are MIT. Their models are downloaded on first use, and each model's license is shown before the download starts.

The five numbers below are targets on the reference machine. None of them has been measured. The reference machine is an RTX 4080 Laptop with 12 GB, 64 GB of RAM, Windows 11 and no admin rights.

- Scrub and play 4K30 HEVC at 30 frames per second or better.
- A slider change on a 24 megapixel photo redraws the preview within 16 milliseconds at display resolution. The full-resolution render finishes within 50 milliseconds.
- Export a 60 second 1080x1920 Reel through NVENC in under 15 seconds.
- Cold start under 2 seconds.
- Memory under 4 GB for a project with ten 4K clips.

Every GPU operator has a CPU twin in gamut-color, and a golden test renders both and compares them within a tolerance. Every shipped template renders frame N against a golden PNG. The fixture corpus is one file per format. It holds a JPEG, a PNG, a HEIC, a gain-map JPEG, an H.264 MP4 and an HEVC MOV. It also holds a 2 second Dolby Vision 8.4 clip, an HLG clip, a PQ clip, one CR3, one ARW and one DNG. The HEIC, HEVC and Dolby Vision files come from the author's phone, the HLG and PQ clips from the platform sample clips, and the CR3, ARW and DNG from public RAW samples. RAW files run 20 to 50 MB each. The fixtures live in a GitHub release asset that CI downloads, outside git. CI runs on windows-latest with the ffmpeg DLLs cached.

Velopack writes a per-user install under LocalAppData with no elevation, and the portable zip is the same exe plus the DLLs it needs. SmartScreen warns on every new app until reputation accumulates, signed or not, so the README says so plainly. SignPath Foundation signs OSI-licensed projects for free, and the publisher users see is "SignPath Foundation".

Version 0.1 starts with these questions unanswered.

- Can the decoded NVDEC frame be shared with wgpu without a system-memory copy on the DX12 backend, and at what version would that land?
- Which subject-mask model ships first, U2-Net at one pass and a smaller size, or MobileSAM with click to refine?
- Does whisper run on the CPU only in version 0.1, or does it get a Vulkan build?
- Is a Lightroom XMP importer worth a milestone, so users can bring presets they already own?
