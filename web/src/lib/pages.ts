import type { CollectionEntry } from 'astro:content';
import { getRelativeLocaleUrl } from 'astro:i18n';
import type { Locale } from '../i18n/ui';

export function pageInfo(entry: CollectionEntry<'pages'>) {
  const locale: Locale = entry.id.endsWith('.ja') ? 'ja' : 'en';
  const slug = entry.id.replace(/\.ja$/, '').replace(/(^|\/)_index$/, '');
  return { locale, slug, path: getRelativeLocaleUrl(locale, slug) };
}
