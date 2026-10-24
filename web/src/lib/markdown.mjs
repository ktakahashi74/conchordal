import { satteri } from '@astrojs/markdown-satteri';
import { htmlToHast } from 'satteri';
import katex from 'katex';
import unidecode from 'unidecode';

// Parse math before CommonMark escapes so TeX such as `\_`, `\,` and `\\` survives.
const katexMath = {
  name: 'katex',
  options: { position: true },
  element: {
    filter: ['pre', 'code'],
    visit(node, ctx) {
      const code = node.tagName === 'pre' ? node.children.find(c => c.tagName === 'code') : node;
      const cls = code?.properties?.className ?? [];
      const block = cls.includes('math-display');
      if (!(block || cls.includes('math-inline')) || (block && node === code)) return;
      // Satteri marks a one-line `$$...$$` as inline; the source delimiter decides.
      const displayMode = block || ctx.source.startsWith('$$', node.position.start.offset);
      // Throw so a TeX error fails the build instead of shipping red source text.
      const options = { displayMode };
      let heading = false;
      for (let p = ctx.parent(node); p && !heading; p = ctx.parent(p)) heading = /^h[1-6]$/.test(p.tagName ?? '');
      // Astro reads heading text after this pass, so heading math stays as text-bearing elements.
      if (heading) return htmlToHast(katex.renderToString(ctx.textContent(code), { ...options, output: 'html' }), { fragment: true }).children[0];
      ctx.replaceNode(node, { type: 'raw', value: katex.renderToString(ctx.textContent(code), options) });
    },
  },
};

// Keep published section URLs and CommonMark parsing.
export const markdown = satteri({
  features: { smartPunctuation: false, math: true },
  mdastPlugins: [{
    name: 'commonmark-links',
    options: { position: true },
    link(node, ctx) {
      const source = ctx.source.slice(node.position.start.offset, node.position.end.offset);
      if (!source.startsWith('[') && !source.startsWith('<')) {
        ctx.replaceNode(node, node.children);
      }
    },
  }],
  hastPlugins: [() => {
    const used = new Set();
    return {
      name: 'published-heading-ids',
      element: {
        filter: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
        visit(node, ctx) {
          const base = unidecode(ctx.textContent(node)).toLowerCase()
            .replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
          let slug = base;
          for (let n = 1; used.has(slug); n++) slug = `${base}-${n}`;
          used.add(slug);
          ctx.setProperty(node, 'id', slug);
        },
      },
    };
  }, katexMath],
});
