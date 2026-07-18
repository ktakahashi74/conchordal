# エディタ設定（LSP）

Conchordalには、名前を持つ呼び出し可能なスクリプティングAPIとhover documentationを
記述したRhai LSP定義ファイルが付属します。対応するclientへ接続すると、`place`、
`harmonic`、`.brain()`、`.send(habitat_bus)`などでcompletion、hover、定義への移動、
diagnosticsを利用できます。busを結合する`|` overloadは名前付き宣言ではなくoperatorなので、
定義ファイルには出力されません。

repository rootにある次の2ファイルが設定の中心です。

- `Rhai.toml` — `rhai-lsp`が読むworkspace設定
- `rhai-defs/conchordal.d.rhai` — doc comment付きの自動生成type/function宣言

upstreamの[rhai-lsp](https://github.com/rhaiscript/lsp)は、自らをexperimentalかつ
incompleteとし、一般利用を推奨していません。保証されたproduction toolingではなく、
任意の開発補助として扱ってください。それでも使う場合は、crates.ioにはないため
git repositoryから直接installします。

```bash
cargo install --git https://github.com/rhaiscript/lsp rhai-cli
```

これにより、`lsp` subcommandを持つ`rhai` binaryが生成されます。

## VS Code

公式[Rhai extension](https://marketplace.visualstudio.com/items?itemName=rhaiscript.vscode-rhai)
が現在提供するのはsyntax highlightだけで、`rhai-lsp`は起動しません。LSP機能には、
Conchordal workspaceから次のcommandを起動できるLSP client/extensionを使います。

```bash
rhai lsp stdio --config Rhai.toml
```

## Neovim 0.11以降

```lua
vim.lsp.config("rhai", {
  cmd = { "rhai", "lsp", "stdio", "--config", "Rhai.toml" },
  filetypes = { "rhai" },
  root_markers = { "Rhai.toml", ".git" },
})
vim.lsp.enable("rhai")
```

## Helix

`~/.config/helix/languages.toml`へ追加します。

```toml
[[language]]
name = "rhai"
scope = "source.rhai"
file-types = ["rhai"]
language-servers = ["rhai-lsp"]

[language-server.rhai-lsp]
command = "rhai"
args = ["lsp", "stdio", "--config", "Rhai.toml"]
```

## Emacs（eglot）

```elisp
(add-to-list 'eglot-server-programs
             '(rhai-mode . ("rhai" "lsp" "stdio" "--config" "Rhai.toml")))
(add-hook 'rhai-mode-hook #'eglot-ensure)
```

## 定義ファイルを再生成する

定義ファイル、英語版[API Reference](/docs/rhai/reference/api.html)、日本語版
[APIリファレンス](../reference/api.md)は、engineの`register_fn` surfaceと
documentation registry（`src/scripting/docs.rs`）を結合して生成されます。
新しいConchordal versionを取得した後にdiagnosticsが欠ける場合は、すべてを再生成します。

```bash
cargo run --bin gen_rhai_defs
```

commit済みartifactが古いとCI testが失敗します。これは登録signatureと生成textを検査しますが、
editor上の挙動はexperimentalなupstream serverとclient設定にも依存します。
