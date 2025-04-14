        if probabilities:
            try:
                prob_df = pd.DataFrame(probabilities)
                # Data pro graf - seřadíme dle DB (nejvyšší první)
                teams_sorted = prob_df['nationality'].tolist()
                win_probs_sorted = prob_df['win_prob'].tolist()

                fig_win = go.Figure(data=[go.Bar(
                    x=teams_sorted,           # Týmy na ose X
                    y=win_probs_sorted,       # Pravděpodobnosti na ose Y
                    name='Výhra v turnaji (%)',
                    marker_color='indianred'
                 )])
                fig_win.update_layout(
                    title='Pravděpodobnost celkového vítězství v turnaji (%)',
                    xaxis_title='Tým',             # Popisek osy X
                    yaxis_title='Pravděpodobnost (%)', # Popisek osy Y
                    xaxis_tickangle=-45         # Natočení popisků osy X
                )
                win_prob_chart_json = json.dumps(fig_win, cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e:
                 logging.error(f"Chyba při generování grafu pravděpodobností: {e}")
                 win_prob_chart_json = "{}"

                         <!--
        {# --- Knockout Bracket --- #}
        <h2>Pavouk Vyřazovací Fáze (první simulace)</h2>
        <div class="knockout-bracket">
            {% for stage, matches in structured_knockout_data.items() %}
                <div class="round">
                    <h3>{{ stage.replace('R16', 'Osmifinále').replace('QF', 'Čtvrtfinále').replace('SF', 'Semifinále').replace('Final', 'Finále') }}</h3>
                    {% for match in matches %}
                        <div class="matchup">
                            {% if match.winner == match.team_a %}
                                <span class="winner">{{ match.team_a }}</span>
                                <span>{{ match.team_b }}</span>
                            {% elif match.winner == match.team_b %}
                                <span>{{ match.team_a }}</span>
                                <span class="winner">{{ match.team_b }}</span>
                            {% else %}
                                <span>{{ match.team_a }}</span>
                                <span>{{ match.team_b }}</span>
                            {% endif %}
                            <span>{{ match.score_a }} - {{ match.score_b }}</span>
                        </div>
                    {% endfor %}
                </div>
            {% endfor %}
        </div>
        -->
