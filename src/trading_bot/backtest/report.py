"""Rapport HTML autonome pour un résultat de backtest.

Génère un unique fichier .html contenant la courbe d'equity, la courbe de
drawdown, la distribution du P&L par trade, et le résumé texte des
métriques. Chaque graphique est encodé en PNG (base64) directement dans le
HTML : aucun JavaScript, aucune dépendance réseau/CDN à l'ouverture, le
fichier s'ouvre hors-ligne dans n'importe quel navigateur.

Nécessite matplotlib, installé via l'extra optionnel `report`
(`pip install -e ".[report]"`) — importé paresseusement ici uniquement, pour
ne pas alourdir les autres commandes du bot qui n'en ont pas besoin.
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import pandas as pd

from trading_bot.backtest.engine import BacktestResult


def generate_html_report(result: BacktestResult, output_path: str | Path, title: str = "Rapport de backtest") -> None:
    """Écrit un rapport HTML autonome décrivant `result` dans `output_path`."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # pas d'affichage interactif : on ne fait que générer des images
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib est requis pour générer un rapport HTML. Installe l'extra dédié : "
            'pip install -e ".[report]"'
        ) from exc

    charts_html = "".join(
        [
            _chart_section(plt, "Courbe d'equity", _plot_equity_curve, result.equity_curve),
            _chart_section(plt, "Drawdown", _plot_drawdown, result.equity_curve),
            _chart_section(plt, "Distribution du P&L par trade", _plot_trade_pnl_histogram, result.trades),
        ]
    )

    html = _render_html(title, result, charts_html)
    Path(output_path).write_text(html, encoding="utf-8")


def _chart_section(plt, heading: str, plot_fn, data) -> str:
    fig = plot_fn(plt, data)
    img_base64 = _fig_to_base64_png(fig, plt)
    return f'<section><h2>{escape(heading)}</h2><img src="data:image/png;base64,{img_base64}" alt="{escape(heading)}"></section>\n'


def _fig_to_base64_png(fig, plt) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_equity_curve(plt, equity_curve: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if len(equity_curve) > 0:
        ax.plot(equity_curve.index, equity_curve.values, color="#2563eb", linewidth=1.5)
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    return fig


def _plot_drawdown(plt, equity_curve: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 3))
    if len(equity_curve) > 0:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1) * 100
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#dc2626", alpha=0.4)
        ax.plot(drawdown.index, drawdown.values, color="#dc2626", linewidth=1.0)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    return fig


def _plot_trade_pnl_histogram(plt, trades: list):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    pnls = [t.pnl for t in trades]
    if pnls:
        colors = ["#16a34a" if p >= 0 else "#dc2626" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Trade # (par ordre chronologique)")
    else:
        ax.text(0.5, 0.5, "Aucun trade réalisé", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("P&L par trade ($)")
    ax.grid(True, alpha=0.3)
    return fig


def _render_html(title: str, result: BacktestResult, charts_html: str) -> str:
    summary_text = escape(result.metrics.summary())
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem auto; max-width: 960px; color: #1f2937; }}
  h1 {{ font-size: 1.5rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; color: #374151; }}
  section img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 6px; }}
  pre {{ background: #f3f4f6; padding: 1rem; border-radius: 6px; white-space: pre-wrap; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<pre>{summary_text}</pre>
{charts_html}
</body>
</html>
"""
