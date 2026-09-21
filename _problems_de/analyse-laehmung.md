---
title: Analyse-Lähmung
description: Teams bleiben in Recherchephasen stecken, ohne zur Umsetzung überzugehen,
  was Fortschritt verhindert.
category:
- Management
- Process
- Team
related_problems:
- slug: decision-paralysis
  similarity: 0.8
- slug: decision-avoidance
  similarity: 0.75
- slug: delayed-decision-making
  similarity: 0.7
- slug: maintenance-paralysis
  similarity: 0.7
- slug: fear-of-failure
  similarity: 0.65
- slug: work-blocking
  similarity: 0.65
solutions:
- architecture-decision-records
- functional-spike
- technical-spike
- walking-skeleton
- decision-rights-and-escalation
- mikado-method
- pilot-projects
- explicit-prioritization-framework
- lightweight-design-review
- small-change-batches
- no-regret-moves
- staged-investment-with-decision-gates
- technical-debt-assessment
- debt-remediation-estimation
- debt-classification
layout: problem
lang: de
en_slug: analysis-paralysis
---

## Description

Analyse-Lähmung entsteht, wenn Entwicklungsteams in endlosen Recherche-, Analyse- und Planungsphasen gefangen bleiben, ohne zur eigentlichen Umsetzungsarbeit überzugehen. Das Team sammelt kontinuierlich Informationen, bewertet Optionen und verfeinert sein Verständnis, fühlt sich aber nie sicher genug, mit dem Bau der Lösung zu beginnen. Diese Lähmung entspringt oft perfektionistischen Tendenzen, der Angst vor falschen Entscheidungen oder dem Fehlen klarer Kriterien dafür, wann die Analyse ausreicht, um mit der Umsetzung fortzufahren.

## Indicators ⟡

- Recherchephasen überschreiten durchgängig ihre geplante Dauer
- Teams verschieben die Umsetzung wiederholt, um mehr Informationen zu sammeln
- Mehrere konkurrierende technische Ansätze werden analysiert, ohne einen davon auszuwählen
- Analysedokumente und Proof-of-Concepts häufen sich an, ohne zu Produktionscode zu führen
- Das Team äußert Unsicherheit darüber, wann es "genug" Informationen hat, um fortzufahren

## Symptoms ▲

- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Verlängerte Recherchephasen verschieben Projektzeitpläne direkt nach hinten, da sich die Starttermine der Umsetzung verzögern.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Teams, die in der Analyse feststecken, produzieren keinen funktionierenden Code, was die Entwicklungsgeschwindigkeit drastisch verringert.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Umfangreiche Analysearbeit, die nie zur Umsetzung führt, stellt verschwendeten Entwicklungsaufwand dar.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Langwierige Analysephasen führen dazu, dass Teams ihre Umsetzungstermine verpassen.
- [Frustration der Stakeholder](frustration-der-stakeholder.md)
<br/>  Stakeholder werden frustriert, wenn Teams Monate mit Analyse verbringen, ohne greifbare Ergebnisse zu liefern.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Geschäftswert kann nicht geliefert werden, solange Teams in Analysephasen feststecken.

## Causes ▼

- [Angst vor Scheitern](angst-vor-scheitern.md)
<br/>  Teams, die Angst haben, falsche technische Entscheidungen zu treffen, analysieren weiter, um das Risiko einer schlechten Entscheidung zu vermeiden.
- [Perfektionismus-Kultur](perfektionismus-kultur.md)
<br/>  Eine Kultur, die perfekte Lösungen vor der Umsetzung verlangt, begünstigt endlose Analyse.
- [Entscheidungslähmung](entscheidungslaehmung.md)
<br/>  Die Unfähigkeit, sich zwischen konkurrierenden Optionen zu entscheiden, lässt Teams weiter Informationen sammeln, in der Hoffnung, die Wahl zu lösen, und verlängert so die Recherchephase; die beiden Probleme verstärken sich häufig gegenseitig in einer Rückkopplungsschleife, statt dass eines das andere rein verursacht.
- [Unklare Ziele und Prioritäten](unklare-ziele-und-prioritaeten.md)
<br/>  Ohne klare Ziele fehlen Teams Kriterien dafür, wann Analyse ausreicht, was zu Überanalyse führt.

## Detection Methods ○

- **Recherchedauer-Tracking:** Beobachtung, wie viel Zeit Teams in Analysephasen verbringen im Vergleich zu geplanten Zeitplänen
- **Entscheidungsprotokoll-Analyse:** Nachverfolgung, wie viele Entscheidungen aufgrund zusätzlicher Analyse aufgeschoben werden
- **Tracking des Umsetzungsstarttermins:** Messung von Verzögerungen zwischen geplantem und tatsächlichem Umsetzungsbeginn
- **Analyseergebnis-Review:** Bewertung, ob Analysedokumente zu umsetzbaren Implementierungsplänen führen
- **Team-Geschwindigkeitsmetriken:** Beobachtung, ob Recherchephasen mit verringerter Entwicklungsgeschwindigkeit korrelieren

## Examples

Ein Entwicklungsteam verbringt vier Monate damit, verschiedene Microservices-Architekturen zu analysieren, zwölf verschiedene Technologien zu bewerten, detaillierte Vergleichsmatrizen zu erstellen und mehrere Proof-of-Concept-Anwendungen zu bauen. Obwohl es bereits nach dem ersten Monat genug Informationen für eine fundierte Entscheidung hätte, analysiert das Team "nur um sicherzugehen" weiter und untersucht Grenzfälle, die in der Praxis möglicherweise nie auftreten. Währenddessen rückt der Projekttermin näher, und es wurde noch kein Produktionscode geschrieben. Ein weiteres Beispiel betrifft ein Team, das sechs Wochen lang Datenbank-Migrationsstrategien recherchiert, aufwendige Testpläne und Performance-Benchmarks erstellt, aber nie tatsächlich mit der Migration beginnt, weil es absolut sicher sein will, jede mögliche Optimierung und Risikominderungsstrategie berücksichtigt zu haben.
