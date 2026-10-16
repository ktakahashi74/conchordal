import { satteri } from '@astrojs/markdown-satteri';
import unidecode from 'unidecode';

// Keep published section URLs and CommonMark parsing; math remains client-side.
export const markdown = satteri({
  features: { smartPunctuation: false, math: false },
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
  }],
});
