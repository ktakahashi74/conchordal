+++
title = "ドキュメント"
description = "心理音響に基づく生成音楽エンジンConchordalの包括的なドキュメント"
sort_by = "weight"
+++

# ドキュメント

## Conchordalでの作曲

Conchordalは**Rhai**スクリプトで制御する。スクリプトでエージェントの種を定義し、生成タイミング、初期パラメータ、ランドスケープの物理パラメータを指定する。

- [シナリオスクリプティングリファレンス](/docs/rhai/)

## 概要

Conchordalの全体像:

- 知覚地形としてのハーモニー:Consonance Field・Density・Movement・Viability を respawn が生態系として閉じる。
- 創発する拍節の上の単一の結合連続体としてのリズム(`metric` / `entrained` / `flow`)。ディレクターはソフトな事前分布で地形を整形する。
- 提示音のみを観測するリスナー側モデル(`ListenerTwin`)と、任意の弱いDCC結合。
- 階層化されたRhaiスクリプティング面(Core / Experimental / Tuning / Research)と、ドリフトしない生成APIリファレンス。

基盤モデルの詳細は[テクニカルノート](/technote/)を参照。
