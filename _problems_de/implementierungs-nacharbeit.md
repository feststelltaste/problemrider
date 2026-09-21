---
title: Implementierungs-Nacharbeit
description: Features müssen neu gebaut werden, wenn sich das anfängliche Verständnis
  als falsch erweist, was Entwicklungsaufwand verschwendet und die Lieferung verzögert.
category:
- Code
- Process
related_problems:
- slug: wasted-development-effort
  similarity: 0.7
- slug: reimplemented-standard-functionality
  similarity: 0.65
- slug: complex-implementation-paths
  similarity: 0.6
- slug: frequent-changes-to-requirements
  similarity: 0.6
- slug: accumulation-of-workarounds
  similarity: 0.6
- slug: feature-creep-without-refactoring
  similarity: 0.55
solutions:
- boring-technologies
- design-by-contract
- functional-spike
- on-site-customer
- prototypes
- prototyping
- specification-by-example
- subject-matter-reviews
- user-acceptance-tests
- user-stories
- behavior-driven-development-bdd
- wireframing
- definition-of-ready
layout: problem
lang: de
en_slug: implementation-rework
---

## Description

Implementierungs-Nacharbeit tritt auf, wenn abgeschlossene Features oder Systemkomponenten erheblich neu gebaut oder neu implementiert werden müssen, weil das anfängliche Verständnis der Anforderungen, technischen Einschränkungen oder des Systemverhaltens falsch war. Diese Nacharbeit stellt verschwendeten Entwicklungsaufwand dar und verlängert Projektzeitpläne, was oft sowohl Entwickler als auch Stakeholder frustriert. Anders als normale iterative Verfeinerung beinhaltet Implementierungs-Nacharbeit fundamentale Änderungen, die mit besserem anfänglichen Verständnis oder Anforderungsanalyse hätten vermieden werden können.

## Indicators ⟡

- Features werden häufig von Grund auf neu gebaut statt schrittweise verbessert
- Abgeschlossene Arbeit wird aufgrund falscher Annahmen über Anforderungen verworfen
- Technische Implementierungen scheitern an Integrationstests aufgrund missverstandener Einschränkungen
- Stakeholder lehnen abgeschlossene Features ab, weil sie den tatsächlichen Bedürfnissen nicht entsprechen
- Entwicklungsschätzungen unterschätzen durchgängig den Bedarf an Nacharbeit

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Das Neubauen von Features, die bereits implementiert waren, verschwendet erhebliche Entwicklungszeit und verzögert die Lieferung.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Arbeit, die verworfen und neu gemacht werden muss, stellt direkte Verschwendung von Entwicklungsressourcen und Teamaufwand dar.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Nacharbeit verdoppelt oder verdreifacht die effektiven Kosten von Features, da sie mehrfach gebaut werden müssen.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Wiederholt verworfene und neu gemachte Arbeit ist demoralisierend und frustrierend für Entwickler.

## Causes ▼

- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Das Bauen von Features basierend auf unvalidierten Annahmen über Anforderungen führt zu Implementierungen, die tatsächliche Bedürfnisse verfehlen.
- [Implementierung beginnt ohne Design](implementierung-beginnt-ohne-design.md)
<br/>  Mit dem Programmieren zu beginnen, ohne ordentliches Design, bedeutet, dass strukturelle Probleme erst spät entdeckt werden, was erhebliches Neubauen erfordert.
- [Anforderungsmehrdeutigkeit](anforderungsmehrdeutigkeit.md)
<br/>  Mehrdeutige oder unvollständige Anforderungen führen zu Fehlinterpretationen, die erst auftauchen, wenn die Implementierung überprüft oder getestet wird.
- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Ohne regelmäßiges Stakeholder-Feedback während der Entwicklung bauen Teams möglicherweise Features, die nicht den tatsächlichen Geschäftsbedürfnissen entsprechen.

## Detection Methods ○

- **Nacharbeits-Tracking:** Beobachtung des Prozentsatzes abgeschlossener Arbeit, die erhebliche Änderungen oder Neubau erfordert
- **Anforderungsänderungsanalyse:** Nachverfolgung, wie oft Anforderungen nach Beginn der Implementierung geklärt oder korrigiert werden
- **Stakeholder-Feedback-Muster:** Analyse von Feedback zur Identifikation wiederkehrender Missverständnismuster
- **Implementierungszyklus-Analyse:** Messung, wie viele Iterationen Features vor der Abnahme benötigen
- **Entwicklerzeit-Analyse:** Nachverfolgung der Zeit, die für Nacharbeit vs. neue Entwicklung aufgewendet wird

## Examples

Ein Entwicklungsteam verbringt drei Wochen mit der Implementierung eines Kundenberichts-Features basierend auf ihrem Verständnis der Anforderungen, nur um während des Nutzertests festzustellen, dass das Berichtsformat nicht den regulatorischen Compliance-Anforderungen entspricht und komplett neu gestaltet werden muss. Das Team hatte den komplexen regulatorischen Kontext nicht verstanden und ein Feature gebaut, das, während funktional korrekt, für seinen beabsichtigten Zweck unbrauchbar war. Ein weiteres Beispiel betrifft ein Team, das eine Performance-Optimierung für eine Datenbankabfrage implementiert, von der es annahm, sie verursache Verlangsamungen, und zwei Wochen mit dem Bau einer komplexen Caching-Schicht verbringt, nur um durch ordentliches Profiling festzustellen, dass der tatsächliche Engpass in einem völlig anderen Teil des Systems lag, was ihre Optimierungsbemühung wertlos machte.
