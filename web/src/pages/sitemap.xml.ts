import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { pageInfo } from '../lib/pages';

export const GET: APIRoute = async ({ site }) => {
  const urls = (await getCollection('pages')).map(entry => new URL(pageInfo(entry).path, site).href).sort();
  return new Response(`<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls.map(url => `
    <url>
        <loc>${url}</loc>
    </url>`).join('')}
</urlset>
`, { headers: { 'Content-Type': 'application/xml' } });
};
