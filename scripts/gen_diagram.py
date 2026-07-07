"""Generate the self-distillation method diagram as inline SVG (theme-aware:
colors come from the page's CSS custom properties)."""

CW, CH, GAP, X0 = 46, 40, 6, 14

# (label, kind, rope_pos)  kinds: ctx, mask1, reg1, mask2, reg2
CELLS = [
    ("x0", "ctx", 0), ("x1", "ctx", 1), ("x2", "ctx", 2),
    ("M", "mask1", 3), ("M", "mask1", 4), ("M", "mask1", 5),
    ("x3", "reg1", 3), ("x4", "reg1", 4), ("x5", "reg1", 5),
    ("x6", "ctx", 6), ("x7", "ctx", 7),
    ("M", "mask2", 8), ("M", "mask2", 9),
    ("x8", "reg2", 8), ("x9", "reg2", 9),
]

FILL = {"ctx": "var(--card)", "mask1": "var(--accent)", "mask2": "var(--accent)",
        "reg1": "none", "reg2": "none"}
STROKE = {"ctx": "var(--rule)", "mask1": "var(--accent)", "mask2": "var(--accent)",
          "reg1": "var(--accent-2)", "reg2": "var(--accent-2)"}
TEXTC = {"ctx": "var(--ink)", "mask1": "var(--paper)", "mask2": "var(--paper)",
         "reg1": "var(--accent-2)", "reg2": "var(--accent-2)"}


def cx(i):
    return X0 + i * (CW + GAP) + CW / 2


def arc(x1, x2, y, h, color, dash="", width=1.6, marker=True):
    m = ' marker-end="url(#arr)"' if marker else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M {x1:.0f} {y} Q {(x1+x2)/2:.0f} {y-h} {x2:.0f} {y}" '
            f'fill="none" stroke="{color}" stroke-width="{width}"{d}{m}/>')


def main():
    W = X0 * 2 + len(CELLS) * (CW + GAP) - GAP
    TOP = 152          # arc space above cells
    YC = TOP           # cell row y
    YP = YC + CH + 14  # rope pos labels
    YD = YP + 26       # distill arrow zone
    H = YD + 80
    p = []
    p.append(f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="masquerade fused training layout" '
             f'style="width:100%;height:auto;font-family:ui-monospace,Menlo,monospace">')
    p.append('<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
             'markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">'
             '<path d="M 0 0 L 8 4 L 0 8 z" fill="context-stroke"/></marker></defs>')

    # cells + labels + rope positions
    for i, (lab, kind, pos) in enumerate(CELLS):
        x = X0 + i * (CW + GAP)
        p.append(f'<rect x="{x}" y="{YC}" width="{CW}" height="{CH}" rx="7" '
                 f'fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="1.6"/>')
        p.append(f'<text x="{cx(i):.0f}" y="{YC + CH/2 + 5}" text-anchor="middle" '
                 f'font-size="15" fill="{TEXTC[kind]}">{lab}</text>')
        p.append(f'<text x="{cx(i):.0f}" y="{YP}" text-anchor="middle" '
                 f'font-size="10.5" fill="var(--muted)">{pos}</text>')
    p.append(f'<text x="{X0}" y="{YP + 15}" font-size="10.5" '
             f'fill="var(--muted)">rope pos</text>')

    A = "var(--accent)"; G = "var(--accent-2)"; MU = "var(--muted)"; R = "var(--warn)"
    y = YC - 4
    # attention arcs from M @ idx 12 (second group, 2nd mask)
    p.append(arc(cx(12), cx(1), y, 96, A))            # -> prior ctx
    p.append(arc(cx(12), cx(7), y, 66, A))            # -> prior region real
    p.append(arc(cx(12), cx(10), y, 38, A))           # -> nearer ctx
    p.append(arc(cx(12), cx(11), y, 20, A, dash="4 3"))  # -> own-group mask
    p.append(f'<text x="{cx(12):.0f}" y="{y-120}" text-anchor="middle" font-size="11" '
             f'fill="{A}">mask attention</text>')
    # blocked: other group's mask
    xm = (cx(12) + cx(4)) / 2
    p.append(arc(cx(12), cx(4), y, 126, R, dash="2 4", width=1.3, marker=False))
    p.append(f'<text x="{xm:.0f}" y="{y-116}" text-anchor="middle" font-size="14" '
             f'fill="{R}">&#10007; other group blocked</text>')
    # teacher stream: real -> real only (small arc, label at right edge)
    p.append(arc(cx(13), cx(10), y, 26, MU, width=1.3))
    p.append(f'<text x="{cx(14)+CW/2:.0f}" y="{y-40}" text-anchor="end" '
             f'font-size="10.5" fill="{MU}">reals attend reals only = teacher stream</text>')

    # distill pairs below: region real -> shadow mask
    yb = YC + CH + 34
    for (ri, mi) in [(6, 3), (7, 4), (8, 5), (13, 11), (14, 12)]:
        p.append(f'<path d="M {cx(ri):.0f} {yb} Q {(cx(ri)+cx(mi))/2:.0f} {yb+34} '
                 f'{cx(mi):.0f} {yb}" fill="none" stroke="{G}" stroke-width="1.6" '
                 f'marker-end="url(#arr)"/>')
    p.append(f'<text x="{cx(5):.0f}" y="{yb + 56}" text-anchor="middle" font-size="11.5" '
             f'fill="{G}">distill: stop-grad teacher logits at x&#8346; &#8594; mask at same rope pos '
             f'(both predict the token after p)</text>')
    p.append('</svg>')
    svg = "\n".join(p)

    s = open("report.html").read()
    START = "<!-- DIAGRAM:START -->"; END = "<!-- DIAGRAM:END -->"
    block = (f'{START}\n<figure style="margin:24px 0">{svg}'
             f'<figcaption>One training forward: k masks are inserted <em>before</em> each '
             f'selected region and carry its rope positions. Blue arcs = mask attention '
             f'(prior real context + earlier masks of its own group; other groups are '
             f'blocked &#10007;). Real tokens never see masks, so the real stream is '
             f'bit-identical to a plain causal forward — it <em>is</em> the live teacher. '
             f'Green arrows = the distillation pairs.</figcaption></figure>\n{END}')
    if START in s:
        pre = s[:s.index(START)]
        post = s[s.index(END) + len(END):]
        s = pre + block + post
    else:
        anchor = '<h2>Method in one figure</h2>'
        s = s.replace(anchor, anchor + "\n  " + block)
    open("report.html", "w").write(s)
    print("diagram inserted", len(svg), "bytes")


if __name__ == "__main__":
    main()
