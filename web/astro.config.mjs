import { defineConfig } from 'astro/config';
import { site } from './src/data/site.ts';
import { locales } from './src/i18n/ui.ts';
import { markdown } from './src/lib/markdown.mjs';

export default defineConfig({
  site: site.url,
  publicDir: './static',
  trailingSlash: 'always',
  i18n: { locales: [...locales], defaultLocale: 'en', routing: { prefixDefaultLocale: false } },
  markdown: {
    processor: markdown,
    syntaxHighlight: false,
  },
});
