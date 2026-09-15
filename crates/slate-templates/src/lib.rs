//! slate-templates owns the template format. A template is a folder holding
//! template.svg and template.toml. The TOML names the text slots, the fonts,
//! the duration and the keyframes as property, time, value and easing. resvg
//! renders static SVG and animates nothing, so all motion lives in the TOML
//! sidecar. Text slots render at runtime and are never baked into the SVG, so
//! wrapping and fonts work.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        // The crate compiles and its test harness runs. Real tests arrive
        // with the milestone that fills the crate.
    }
}
