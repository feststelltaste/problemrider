---
title: Konkurrierende Prioritäten
description: Mehrere dringende Projekte oder Initiativen konkurrieren um dieselben
  begrenzten Ressourcen, was Konflikte und Ineffizienzen erzeugt.
category:
- Management
- Performance
related_problems:
- slug: priority-thrashing
  similarity: 0.7
- slug: unclear-goals-and-priorities
  similarity: 0.65
- slug: project-resource-constraints
  similarity: 0.65
- slug: team-confusion
  similarity: 0.65
- slug: product-direction-chaos
  similarity: 0.65
- slug: power-struggles
  similarity: 0.6
solutions:
- technical-debt-backlog
- error-budgets
- explicit-prioritization-framework
- capacity-based-planning
- work-in-progress-limits
- product-owner
- improvement-budget
- outcome-based-goal-setting
- value-stream-mapping
- feature-usage-measurement
- cost-of-delay
- executive-sponsorship
- value-hierarchy
- debt-classification
- customization-cost-attribution
layout: problem
lang: de
en_slug: competing-priorities
---

## Description

Konkurrierende Prioritäten entstehen, wenn mehrere Projekte, Initiativen oder Aufgaben alle als dringend oder kritisch eingestuft werden, was Teams zwingt, ihre Aufmerksamkeit und Ressourcen zwischen widersprüchlichen Anforderungen aufzuteilen. Dies schafft eine Situation, in der keine einzelne Priorität ausreichend Fokus erhält, was zu suboptimalen Ergebnissen über alle Initiativen hinweg führt. Teams werden überdehnt, Kontextwechsel nehmen zu, und die Organisation schafft es nicht, bei irgendeiner Priorität sinnvollen Fortschritt zu machen.

## Indicators ⟡

- Mehrere Projekte werden gleichzeitig als "oberste Priorität" oder "kritisch" bezeichnet
- Teammitglieder sind mehreren hochprioren Initiativen gleichzeitig zugewiesen
- Ressourcen müssen über konkurrierende dringende Projekte hinweg geteilt werden
- Termine für unterschiedliche Prioritäten stehen im Konflikt zueinander
- Das Management kann nicht klar artikulieren, welche Priorität Vorrang haben soll

## Symptoms ▲

- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Entwickler wechseln ständig zwischen konkurrierenden Projekten und verlieren Produktivität durch Kontextwechsel.
- [Unvollständige Projekte](unvollstaendige-projekte.md)
<br/>  Kein Projekt erhält ausreichenden Fokus, um zur Fertigstellung zu gelangen, wenn Ressourcen über zu viele Prioritäten verteilt sind.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Das Aufteilen der Aufmerksamkeit auf mehrere dringende Initiativen verringert die Gesamtleistung des Teams über alle Projekte hinweg.
- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Alle konkurrierenden Projekte erleben Verzögerungen, weil keines die fokussierte Aufmerksamkeit erhält, die für eine rechtzeitige Fertigstellung nötig ist.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Über Prioritäten hinweg überdehnt, machen Teams Abkürzungen und senken Qualitätsstandards, um bei allem Fortschritt zu machen.
- [Demoralisierung des Teams](demoralisierung-des-teams.md)
<br/>  Die Unfähigkeit, bei irgendeiner Priorität sinnvollen Fortschritt zu machen, demoralisiert Teams, die das Gefühl haben, dass ihre Anstrengungen vergeblich sind.

## Causes ▼

- [Unklare Ziele und Prioritäten](unklare-ziele-und-prioritaeten.md)
<br/>  Fehlende klare organisatorische Ausrichtung führt dazu, dass mehrere Initiativen als gleich dringend behandelt werden.
- [Machtkämpfe](machtkaempfe.md)
<br/>  Unterschiedliche Abteilungen oder Manager konkurrieren darum, ihre Projekte priorisiert zu bekommen, was widersprüchliche Anforderungen an gemeinsame Ressourcen schafft.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Unzureichende Planung versäumt es, Initiativen angemessen zu sequenzieren, was zu Überlappungen mehrerer kritischer Projekte führt.
- [Marktdruck](marktdruck.md)
<br/>  Externe Wettbewerbskräfte schaffen gleichzeitig wirklich dringende Anforderungen an mehreren Fronten.

## Detection Methods ○

- **Prioritätenzähl-Analyse:** Nachverfolgung, wie viele Initiativen gleichzeitig als oberste Priorität bezeichnet werden
- **Ressourcenzuweisungs-Mapping:** Visualisierung, wie Ressourcen über konkurrierende Prioritäten aufgeteilt werden
- **Team-Zuweisungsüberlappung:** Messung, wie vielen hochprioren Projekten Teammitglieder zugewiesen sind
- **Kontextwechsel-Häufigkeit:** Beobachtung, wie oft Teammitglieder zwischen unterschiedlichen Prioritäten wechseln
- **Stakeholder-Prioritätsumfragen:** Vergleich des Verständnisses unterschiedlicher Stakeholder von organisatorischen Prioritäten

## Examples

Ein Entwicklungsteam ist gleichzeitig drei "kritischen" Projekten zugewiesen: der Modernisierung des Authentifizierungssystems (aufgrund von Sicherheitslücken), der Umsetzung eines neuen Kundenportals (Schlüsselkunden versprochen) und dem Upgrade des Zahlungsabwicklungssystems (für regulatorische Compliance erforderlich). Jedes Projekt hat unterschiedliche Stakeholder, die ihr Projekt für das wichtigste halten, und jedes hat dringende Termine. Das Team verbringt Montag mit der Authentifizierung, Dienstag mit dem Kundenportal und Mittwoch mit der Zahlungsabwicklung, verliert erhebliche Zeit durch Kontextwechsel und gewinnt bei keinem einzelnen Projekt genug Schwung, um sinnvollen Fortschritt zu machen. Alle drei Projekte verzögern sich am Ende, und die Qualität leidet, weil das Team sich nicht tief auf die Lösung komplexer Probleme konzentrieren kann. Ein weiteres Beispiel betrifft ein Start-up, bei dem der CEO erklärt, dass Nutzerakquise, Produktentwicklung und technische Infrastruktur alle gleich kritische Prioritäten sind. Das kleine Entwicklungsteam versucht, den Bau neuer Features, das Beheben technischer Schulden und die Umsetzung von Wachstumsexperimenten auszubalancieren, aber der ständige Prioritätenwechsel bedeutet, dass es bei keinem Bereich minimalen Fortschritt macht. Das Produkt wird zunehmend fehlerhaft, die Nutzerakquise stagniert aufgrund schlechter Nutzererfahrung, und Infrastrukturprobleme verschlimmern sich, weil sie nie umfassend angegangen werden.
