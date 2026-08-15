# Mistakes Log — Analysis Workbench Session

A record of regressions introduced (or uncovered) while iterating on the Analysis
Workbench UI in one long session, why they happened, and the rule that should
prevent each one from recurring. The common thread: small, well-intentioned CSS/JS
changes had side effects that only showed up live in a browser, which nobody
verified before calling the change done — because no headless browser was available
in this environment.

## 1. "Fix the scrollbar position" broke the footer and article link clicks

**What happened:** The article column's native scrollbar sat at the outer edge of
the browser window instead of next to the workbench panel. The fix turned `body`
into a flex column (header / article / footer as flex siblings) so the article
could scroll in its own box. This pinned the footer permanently visible right
below the article instead of only appearing after scrolling a long article to its
true end — the footer visually ended up "on the left side, out of place." Clicking
links inside the article also broke as a side effect of the same restructuring.

**Root cause:** The footer is a DOM sibling of the article `<main>`, not nested
inside it. Making `body` a flex column with the article as the only `flex-grow`
item necessarily turns the footer into a fixed-size sibling item that renders
immediately — there is no way to keep the old "only after scrolling all the way
down" behavior without literally nesting the footer inside the same scroll
container, which needs a layout template change, not just CSS.

**Rule:** Before changing how a *shared, cross-page* layout element (header, footer,
`body` itself) is positioned, trace every page state that depends on its current
behavior (mobile, expanded workbench, plain article) — not just the one state the
request is about. Prefer a revert over layering a second fix on top of a
layout change you cannot see rendered.

## 2. The revert-and-retry cost multiple round trips because nothing was verified live

**What happened:** The scrollbar fix, its footer regression, and the subsequent
revert all happened via direct source edits and `node --check`/Sass-compile checks
only. No screenshot or live click test ever confirmed the actual visual result at
any point — every one of these bugs was reported by the user after the fact, not
caught before shipping.

**Root cause:** No headless browser (`chromium-cli`, Playwright, etc.) was
available in this container, and syntax-level checks (`node --check`, `jekyll
build`) only prove the code parses and compiles — they say nothing about whether
`position: fixed`, `overflow`, `z-index`, or flex sizing actually render the way
intended.

**Rule:** Treat "compiles" and "looks correct" as different claims. When no way to
render the page exists, say so explicitly and flag every layout-affecting change
as unverified, instead of reporting it as done.

## 3. Hover text went invisible (white-on-white) from a CSS specificity conflict

**What happened:** A reference row's title button was styled with a transparent
background and a deep-blue hover color. On hover the text disappeared instead —
it turned white with no background, unreadable.

**Root cause:** A pre-existing generic rule, `.analysis-trail__node-menu-list
button:hover { color: #fff; background: #007acc; }`, has *higher* specificity
than a plain single-class selector like `.analysis-trail__node-menu-list-title:hover`
because it also matches on the `button` tag. The new rule's `background:
transparent !important` won (importance beats specificity), but its `color` had no
`!important`, so the *old* rule's white text still applied — on top of the new
transparent background. Two rules "merged" per-property instead of one winning
outright, and the merge was invisible until actually hovered in a browser.

**Rule:** When overriding one specific state (hover/focus) of a shared, generic
selector, override *every* property that selector sets for that state, not just the
one that differs — otherwise the cascade quietly recombines old and new
declarations per-property.

## 4. `element.hidden = true` did nothing because a custom `display` rule beat it

**What happened:** A "filter the list locally" feature set `row.hidden = true` on
rows that didn't match a query. The rows stayed visible regardless of the filter.

**Root cause:** The browser's built-in `[hidden] { display: none }` rule lives in
the user-agent stylesheet, which loses to *any* author-stylesheet rule of equal or
lower specificity that also sets `display` — including
`.analysis-trail__node-menu-list-item { display: flex; }`, added earlier in the
same feature for the row's layout. Setting the `hidden` attribute on an element
that also has an explicit `display` in project CSS is silently a no-op.

**Rule:** Never rely on the `hidden` attribute for an element whose class already
sets `display` elsewhere. Either toggle the class instead, or add an explicit
`&[hidden] { display: none !important; }` next to the `display` declaration that
would otherwise cancel it. (This feature was later removed entirely, but the same
trap will resurface anywhere else `.hidden` is set on a styled element.)

## 5. A sensible-looking default ("local" search scope) looked like a total feature failure

**What happened:** A local/global search-scope toggle defaulted to "local," which
only filters the handful of items already listed on screen. Since most nodes have
few or zero pre-linked references, typing into the search box appeared to do
nothing at all — reported as "the search isn't working anymore," not "the default
is wrong."

**Root cause:** The feature worked exactly as coded; the default made the common
case look broken. Later the whole local/global distinction was scrapped as
unnecessary complexity — the "local" list was already visible below, so the search
box only ever needed to do one thing: search the full catalog.

**Rule:** When a control changes what a *different* control (here: the search box)
does, the default matters as much as the logic. If a default makes the most common
first interaction look like a no-op, that's a bug in the default, not just a
preference to tune later. Simpler is often correct: this whole toggle turned out to
be unnecessary.

## 6. A fixed-position element silently ate clicks meant for something behind it — only on some pages

**What happened:** A "Speed nav" checkbox was added to the site header's nav
`.trigger` row, right before the "Analysis Workbench" button. It worked on most
pages but could not be clicked on problem/solution article pages specifically.

**Root cause:** The Analysis Workbench button only renders on problem/solution
pages (gated by `page.collection`) and is `position: fixed`, floating at a fixed
viewport position independent of normal document flow. Placed as a flow neighbor
in the same nav row, it could visually overlap the new checkbox's flow position on
exactly the page type where it exists — invisible in the markup, invisible in the
CSS, only reproducible by actually clicking in a browser on that specific page
type.

**Rule:** A `position: fixed` element floating over a specific page type is
invisible to "does this compile / does the DOM look right" review. Any new
clickable control added near one must be checked specifically on the pages where
the fixed element exists, not just "on the site" in general. When in doubt, move
the new control away from the fixed element's screen region entirely rather than
trying to make them coexist in the same row.

## 7. A years-old scope bug only surfaced once someone actually clicked an in-article link

**What happened:** Clicking a plain link inside article content threw `Uncaught
ReferenceError: loadPageDynamically is not defined`. This was not a regression
from this session — an AST-level scope trace confirmed it already existed before
any of this session's changes.

**Root cause:** `loadPageDynamically` (and the shared `node` variable it updates)
was declared *nested inside* `render()`, while the article-link click handler that
called it lived in a sibling closure (the page-init `DOMContentLoaded` handler) with
no access to render()'s locals. Every call site *inside* `render()` worked fine;
the one call site outside it was silently broken from the day it was written.

**Rule:** A `ReferenceError` on a function that "is clearly defined in this file"
is a scope bug, not a typo — check with an actual parse/AST trace (`acorn` or
similar) which enclosing function a definition and its call site are really nested
in, rather than trusting indentation or assuming shared scope between two
`document.addEventListener` blocks in the same file. Shared mutable state that
several independent closures need (like "the current node") belongs at module
scope from the start, not local to whichever closure happened to need it first.

## 8. Rebuilt CSS/JS still showed the old, broken behavior in the browser

**What happened:** After fixing the `loadPageDynamically` scope bug and rebuilding,
the user hit the *exact same* error, at the *exact same* `?v=...` asset URL, as
before the fix.

**Root cause:** `bundle exec jekyll build --incremental` recompiles changed
SCSS/JS files themselves, but does not regenerate HTML pages whose only
relationship to those assets is a `<script src="...?v={{ site.time }}">` tag —
Jekyll's incremental dependency graph doesn't track that link. The already-built
HTML kept referencing the *old* timestamp, so the browser kept serving its cached
copy of the old, broken script from that exact URL, even though the file on disk
was already fixed.

**Rule:** `--incremental` is for fast iteration and syntax-level sanity checks
only. Before asking anyone to verify a JS/CSS change live in a browser, run one
plain `bundle exec jekyll build` to force every page's cache-busting timestamp to
update. (Now documented in `CLAUDE.md`.)

## 9. A new non-article page silently broke nav clicks — from *and* to it — via a click handler that assumed every page (both ends) is an article

**What happened:** Clicking "Problems" or "Solutions" in the site nav did
nothing at all — no navigation, no error — from two different starting
points: from the new `/landscape/` page (or home, categories, any non-article
page), and *also* from a perfectly normal problem/solution article page.

**Root cause:** `analysis-trail.js` has a document-wide `click` listener that
hijacks any link whose path matches `/\.html$/` or `/\/(problems|solutions)\//`
so it can load it into `.page-main-content` without a full page reload. That
path test is a plain substring match, so it matches the `/problems/` and
`/solutions/` *listing* pages just as readily as a link to one specific
article. The listing pages use the plain `{{ content }}` layout, not the
`problem`/`solution` layout, so they never render a `.page-main-content`
wrapper. That breaks the hijack on *either* side of the navigation:
- Clicked *from* a page without that wrapper (landscape, home, ...): the
  handler still calls `event.preventDefault()`, but `loadPageDynamically`
  bails out immediately since the *current* page has nothing to swap into.
- Clicked *from* an article *to* the listing page: `loadPageDynamically` gets
  as far as fetching the target, but finds no `.page-main-content` in *that*
  response either, so it resets and quietly gives up.

The first fix attempt only guarded the current-page case, which fixed
landscape → nav-click but left article → nav-click broken — same underlying
overly-broad path test, just tripping the failure from the other direction.

**Rule:** A "hijack this link and load it dynamically" handler needs the
wrapper element on *both* ends of the navigation — the current page (to swap
into) and the fetched target (to swap in) — not just one. The robust fix is
narrowing the path test itself to exactly what the handler can actually
handle (here: `/\/(problems|solutions)\/[^/]+\.html$/`, real articles only),
rather than guarding one direction and assuming the other is fine. When a
"only fetch this if X" check and a "does X actually exist here" check are two
different regexes/conditions in two different functions, expect them to
drift out of sync — check both directions before calling a fix complete.

---

**Overall pattern:** almost every entry above is a change that looked correct in
the source and compiled cleanly, but only failed once actually rendered, clicked,
or hovered in a real browser — and no real browser was available to check with in
this environment. Treat any layout, hover-state, or click-target change as
unverified until someone (or something) has actually looked at it rendered, and
say so plainly instead of reporting the change as finished.
