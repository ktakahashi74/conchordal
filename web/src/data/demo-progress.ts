// Source of truth: web/content/technote.md §9.1 and technote.ja.md §9.1.
// Voice research and browser scope are described outside this table.
export const progress = [
  { stage: ['Implemented', '実装済み'], title: ['Frequency terrain', '周波数の地形'], text: ['Roughness, harmonicity and consonance shape the field heard by the ecology.', '粗さ・調波性・協和性が、生態の聴く地形を作ります。'] },
  { stage: ['Partial', '部分'], title: ['Beat and participation', '拍と参加'], text: ['Auditory recurrence drives a shared meter and local participation. Persistence across all conditions remains open.', '音の再来から共有の拍を捉え、各個体が発音への参加を決めます。全条件での時間関係の持続は未解決です。'] },
  { stage: ['Partial', '部分'], title: ['Auditory temporal structure', '聴覚に基づく時間構造'], text: ['Auditory timing mechanisms are partly implemented; phrase perception and scene expectation remain open. The composer arranges this draft’s sections.', '聴覚に基づく時間の機構は一部を実装しています。句の知覚と区間の期待は未解決で、この試作の区間構成は試作者が決めています。'] },
  { stage: ['Implemented', '実装済み'], title: ['Flora: spectral ecology', 'Flora：スペクトルの生態'], text: ['Consumer–resource dynamics, timing and continuous amplitude targets operate as author rules. The API is experimental.', '消費者と資源の動態、発音の時機、連続した振幅の目標を作者規則として実装しています。API は実験段階です。'] },
  { stage: ['Partial', '部分'], title: ['Adaptation and expectation', '順応と期待'], text: ['Habituation, short-term recurrence and bounded harmonic-relation memory are present. Phrase/scene expectation and action-value learning remain open.', '慣れ、短期の反復予測、演奏内の有界な和声関係の記憶を実装しています。句・区間の期待と行動価値学習は未解決です。'] },
  { stage: ['Open', '未解決'], title: ['Biosignal closed loop', '聴き手の生体信号との閉ループ'], text: ['A simulated listener loop exists and is off by default. A physical biosignal loop is not implemented in this demo.', '模擬的な聴き手とのループは実装済みで、既定では無効です。このデモは実際の生体信号を使いません。'] },
  { stage: ['Partial', '部分'], title: ['Performance without an end', '終わりのない演奏'], text: ['Native supports endless scores within verified operating ranges. Browser does not support them; Repeat starts a new finite performance.', 'パソコン用アプリは検証済みの範囲で終端のない台本を実行できます。ブラウザー版は未対応です。「繰り返す」は有限の演奏を新しく始めます。'] },
];
