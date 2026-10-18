export const site = {
  url: 'https://www.conchordal.org',
  title: 'Conchordal',
  tagline: 'A living generative music instrument for emergent harmony',
  author: 'Koichi Takahashi / Conchordal',
  authorName: '髙橋恒一 (Koichi Takahashi)',
  contact: { name: '髙橋恒一', email: 'contact@conchordal.org' },
  image: '/screenshot1.png',
  navigation: [
    { key: 'home', slug: '' },
    { key: 'manifesto', slug: 'manifesto' },
    { key: 'technote', slug: 'technote' },
    { key: 'docs', slug: 'docs' },
  ],
  social: [
    { key: 'github', label: 'GitHub', url: 'https://github.com/ktakahashi74/conchordal', icon: 'github' },
    { key: 'x', label: 'X', url: 'https://x.com/conchordal', icon: 'x' },
  ],
  roadmap: [
    { date: 'c. 1994', text: 'Core concept conceived' },
    { date: 'Aug 2025', text: 'Project started' },
    { date: 'Dec 2025', text: 'v0.1.0 source & web release (pre-alpha)' },
    { date: 'Mar 2026', text: 'v0.3.0 pre-alpha — ', link: { url: 'https://arxiv.org/abs/2603.25637', text: 'paper on arXiv' } },
    { date: 'Aug 2026', text: 'v0.4.0 Alpha — ', link: { url: 'https://2026.alife.org/wp-content/uploads/sites/2/2026/08/ALIFE2026-Program-Semifinal-2026-08-10-Rev5-1.pdf#page=8', text: 'ALIFE 2026 oral presentation' } },
    { date: 'Oct 2026', text: 'v0.5.0 Alpha — sustained articulation, adaptive meter, unified self-exclusion, and runtime/API hardening — ', state: 'current', link: { url: 'https://github.com/ktakahashi74/conchordal/releases/tag/v0.5.0', text: 'release notes' } },
    { date: '2026–2027', text: 'Beta with first compositions', state: 'future' },
  ],
} as const;
