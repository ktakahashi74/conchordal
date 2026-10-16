import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

export const collections = {
  pages: defineCollection({
    loader: glob({ pattern: '**/*.md', base: './content', generateId: ({ entry }) => entry.replace(/\.md$/, '') }),
    schema: z.object({
      title: z.string(),
      description: z.string().optional(),
      date: z.coerce.date().optional(),
      extra: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).default({}),
    }),
  }),
};
