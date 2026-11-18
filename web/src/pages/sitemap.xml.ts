import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { pageInfo } from '../lib/pages';
import { demo } from '../lib/demo';

export const GET: APIRoute = async ({ site }) => {
  const paths = (await getCollection('pages')).map(entry => pageInfo(entry).path);
  if (demo()) paths.push('/demo/', '/ja/demo/');
  const urls = paths.map(path => new URL(path, site).href).sort();
  return new Response(`<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls.map(url => `
    <url>
        <loc>${url}</loc>
    </url>`).join('')}
</urlset>
`, { headers: { 'Content-Type': 'application/xml' } });
};
