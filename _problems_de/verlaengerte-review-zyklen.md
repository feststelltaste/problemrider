---
title: Verlängerte Review-Zyklen
description: Code-Reviews erfordern mehrere Runden von Feedback und Überarbeitung,
  was die Zeit von der Code-Einreichung bis zur Freigabe erheblich verlängert.
category:
- Process
- Team
related_problems:
- slug: code-review-inefficiency
  similarity: 0.8
- slug: reduced-code-submission-frequency
  similarity: 0.75
- slug: large-pull-requests
  similarity: 0.75
- slug: inadequate-initial-reviews
  similarity: 0.75
- slug: rushed-approvals
  similarity: 0.75
- slug: extended-cycle-times
  similarity: 0.7
solutions:
- code-review-process-reform
- small-change-batches
- code-review-guidelines
- work-in-progress-limits
- trunk-based-development
- pair-and-mob-programming
- team-working-agreements
- value-stream-mapping
- delivery-performance-metrics
- fast-feedback-loops
layout: problem
lang: de
en_slug: extended-review-cycles
---

## Description

Verlängerte Review-Zyklen entstehen, wenn Code-Reviews mehrere Runden von Feedback und Überarbeitung erfordern, bevor sie genehmigt werden, was die Zeit von der ursprünglichen Code-Einreichung bis zur endgültigen Akzeptanz erheblich verlängert. Während etwas Überarbeitung normal und gesund ist, beinhalten verlängerte Zyklen übermäßiges Hin und Her, das abnehmende Erträge für die Codequalität bringt, während es erheblich Entwicklerzeit verbraucht und Verzögerungen bei der Feature-Lieferung schafft.

## Indicators ⟡

- Code-Reviews erfordern regelmäßig 4 oder mehr Überarbeitungsrunden
- Einfache Änderungen brauchen Tage oder Wochen, um genehmigt zu werden
- Review-Kommentare identifizieren in späteren Runden weiterhin neue Probleme, die ursprünglich hätten entdeckt werden können
- Autoren verbringen mehr Zeit mit der Bearbeitung von Review-Feedback als mit dem Schreiben des ursprünglichen Codes
- Genehmigungszeiten für Reviews variieren dramatisch bei Änderungen ähnlicher Komplexität

## Symptoms ▲

- [Verlängerte Durchlaufzeiten](verlaengerte-durchlaufzeiten.md)
<br/>  Mehrere Review-Runden blähen direkt die Gesamtzeit von der Code-Einreichung bis zur Produktionslieferung auf.
- [Verringerte Häufigkeit von Code-Einreichungen](verringerte-haeufigkeit-von-code-einreichungen.md)
<br/>  Entwickler bündeln Änderungen, um häufige, schmerzhafte Review-Zyklen zu vermeiden, was die Integrationshäufigkeit verringert.
- [Frustration der Autoren](frustration-der-autoren.md)
<br/>  Entwickler werden frustriert, wenn einfache Änderungen viele Überarbeitungsrunden erfordern, und haben das Gefühl, ihre Zeit werde für abnehmende Erträge verschwendet.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Features und Fixes, die bereits umgesetzt sind, verbleiben über längere Zeit im Review, bevor sie Nutzer erreichen.
- [Vermeidung des Review-Prozesses](vermeidung-des-review-prozesses.md)
<br/>  Langwierige und schmerzhafte Review-Zyklen motivieren Entwickler, Wege zu finden, den Review-Prozess zu umgehen oder zu minimieren.
- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Mehrere Review-Runden zwingen Autoren dazu, wiederholt zu Code zurückzuwechseln, den sie vor Tagen oder Wochen geschrieben haben, wobei sie jedes Mal den Kontext verlieren.

## Causes ▼

- [Perfektionistische Review-Kultur](perfektionistische-review-kultur.md)
<br/>  Eine Kultur, die durch Reviews Perfektion verlangt, führt zu endlosen Runden von Kleinigkeitskritik, statt ausreichend guten Code zu akzeptieren.
- [Unzureichende Erst-Reviews](unzureichende-erst-reviews.md)
<br/>  Oberflächliche Erstrunden-Reviews, die wichtige Probleme übersehen, zwingen nachfolgende Runden dazu, das aufzufangen, was früher hätte gefunden werden sollen.
- [Widersprüchliche Reviewer-Meinungen](widerspruechliche-reviewer-meinungen.md)
<br/>  Unterschiedliche Reviewer, die widersprüchliches Feedback liefern, zwingen Autoren durch zusätzliche Runden, um gegensätzliche Anleitungen zu vereinen.
- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne vereinbarte Coding-Standards bringt jede Review-Runde neue stilistische Präferenzen unterschiedlicher Reviewer zum Vorschein.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Große Pull Requests sind schwerer in einem Durchgang gründlich zu überprüfen, was dazu führt, dass Probleme über mehrere Runden hinweg entdeckt werden.

## Detection Methods ○

- **Review-Runden-Tracking:** Beobachtung der Anzahl der Überarbeitungsrunden, die für unterschiedliche Arten von Änderungen erforderlich sind
- **Review-Dauer-Analyse:** Messung der Gesamtzeit von der Einreichung bis zur Genehmigung für unterschiedliche Änderungsgrößen
- **Feedback-Qualitätsbewertung:** Analyse, ob frühe Review-Runden die wichtigsten Probleme abfangen
- **Zeitinvestition der Autoren:** Nachverfolgung, wie viel Zeit Entwickler für Review-Überarbeitungen im Vergleich zu neuer Entwicklung aufwenden
- **Review-Effizienzmetriken:** Vergleich von Review-Zyklen über unterschiedliche Teams oder Reviewer hinweg

## Examples

Ein Entwickler reicht eine 200-Zeilen-Feature-Implementierung ein, die über drei Wochen sechs Review-Runden durchläuft. Die erste Runde konzentriert sich auf Codestil, die zweite auf Fehlerbehandlung, die dritte auf Performance-Bedenken, die vierte auf den Testansatz, die fünfte auf Variablenbenennung und die sechste auf Dokumentation. Jede Runde erfordert 1-2 Tage Arbeit des Autors und 1-2 Tage Reviewer-Durchlaufzeit. Bis der Code genehmigt ist, hat der Autor den Kontext zur ursprünglichen Implementierung verloren, und die Feature-Lieferung ist um einen Monat verzögert. Ein weiteres Beispiel betrifft eine einfache Fehlerbehebung, die vier Review-Runden erfordert, weil unterschiedliche Reviewer in jeder Runde unterschiedliche Aspekte zur Verbesserung identifizieren – zuerst den Behebungsansatz, dann die Testabdeckung, dann die Fehlermeldungen und schließlich das Logging. Die Behebung, die einen Tag hätte dauern sollen, verbraucht letztlich eine Woche Kalenderzeit und mehrere Stunden Entwicklungsaufwand über das Team hinweg.
