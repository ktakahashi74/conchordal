import { site } from '../data/site';

export const locales = ['en', 'ja'] as const;
export type Locale = (typeof locales)[number];

interface UiStrings {
  description: string;
  keywords: string;
  navigation: Record<'home' | 'manifesto' | 'technote' | 'docs', string>;
  language: string;
  menu: string;
  quickStart: string;
  toc: string;
  backToContents: string;
  provenance: string;
  mathRegion: string;
  tableRegion: string;
  skip: string;
  screenshotAlt: string;
  edit: string;
  sourceBefore: string;
  sourceAfter: string;
}

export const ui: Record<Locale, UiStrings> = {
  en: {
    description: site.tagline,
    keywords: 'generative music instrument, generative music, algorithmic composition, psychoacoustics, consonance, emergent harmony, music cognition, artificial life, autonomous agents, open source, direct cognitive coupling, rust',
    navigation: { home: 'Home', manifesto: 'Manifesto', technote: 'Technote', docs: 'Docs' },
    language: '日本語', menu: 'Menu', quickStart: 'Quick Start →', toc: 'Table of Contents',
    backToContents: 'Back to contents', provenance: 'Provenance and page details',
    mathRegion: 'Scrollable equation', tableRegion: 'Scrollable table', skip: 'Skip to main content',
    screenshotAlt: 'Conchordal instrument interface showing audio meters, auditory attention and neural activity traces, four agents across a frequency axis, and intensity, consonance, roughness and harmonicity plots.',
    edit: 'Edit this page', sourceBefore: 'Source code available on ', sourceAfter: '.',
  },
  ja: {
    description: '直接認知結合による生成音楽システム',
    keywords: '生成音楽, 生成音楽楽器, アルゴリズム作曲, 心理音響, 協和, 創発する調和, 音楽認知, 人工生命, 自律エージェント, オープンソース, 直接認知結合, Rust',
    navigation: { home: 'Home', manifesto: 'Manifesto', technote: 'Technote', docs: 'Docs' },
    language: 'EN', menu: 'Menu', quickStart: 'クイックスタート →', toc: '目次',
    backToContents: '目次に戻る', provenance: '出典・ページ情報',
    mathRegion: '横スクロールできる数式', tableRegion: '横スクロールできる表', skip: '本文へ移動',
    screenshotAlt: 'Conchordalの楽器画面。音量メーター、聴覚的注意と神経活動の時系列、周波数軸上の4つのエージェント、音の強度・協和・粗さ・調波性のグラフを表示している。',
    edit: 'このページを編集', sourceBefore: 'ソースコードは ', sourceAfter: ' で公開されています。',
  },
};
