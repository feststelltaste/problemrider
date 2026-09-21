---
title: Value Stream Mapping
description: Abbildung jedes Schritts von der Anfrage bis zur Produktion
  mit Bearbeitungs- und Wartezeit, sodass die Wartezeit — die fast immer
  den Großteil ausmacht — sichtbar und angehbar wird.
category:
- Process
- Management
problems:
- inefficient-processes
- increased-manual-work
- wasted-development-effort
- extended-cycle-times
- long-release-cycles
- delayed-value-delivery
- operational-overhead
- increased-time-to-market
- approval-dependencies
- work-blocking
- immature-delivery-strategy
- resource-waste
- delayed-project-timelines
- budget-overruns
- cascade-delays
- constantly-shifting-deadlines
- context-switching-overhead
- delayed-issue-resolution
- maintenance-cost-increase
- missed-deadlines
- organizational-structure-mismatch
- project-resource-constraints
- reduced-team-productivity
- team-coordination-issues
- unrealistic-deadlines
- bottleneck-formation
- competing-priorities
- extended-review-cycles
- increased-stress-and-burnout
- mental-fatigue
- planning-credibility-issues
- planning-dysfunction
- priority-thrashing
- process-design-flaws
- team-demoralization
- uneven-work-flow
- uneven-workload-distribution
- process-software-misfit
layout: solution
lang: de
en_slug: value-stream-mapping
related_solutions:
- slug: delivery-performance-metrics
  similarity: 0.7
- slug: work-in-progress-limits
  similarity: 0.7
- slug: impact-mapping
  similarity: 0.65
- slug: story-mapping
  similarity: 0.65
- slug: fast-feedback-loops
  similarity: 0.65
- slug: decision-rights-and-escalation
  similarity: 0.65
---

## Description

Value Stream Mapping protokolliert jeden Schritt, den ein Arbeitsstück von der Anfrage bis zur Produktion durchläuft, und erfasst für jeden Schritt zwei Zahlen: wie lange aktiv an der Arbeit gearbeitet wird, und wie lange sie wartet. Das Verhältnis zwischen ihnen ist der Befund. Teams schätzen konsistent, dass der Großteil ihrer Zykluszeit Entwicklungsaufwand ist; die Karte zeigt konsistent, dass achtzig bis fünfundneunzig Prozent Warten ist — auf Review, auf Genehmigung, auf eine Testumgebung, auf ein Release-Fenster, auf ein anderes Team. Dies zählt, weil Verbesserungsaufwand üblicherweise auf die Bearbeitungszeit zielt, wo die potenziellen Gewinne klein und die Störung hoch sind, während das Warten ist, wo die Zeit tatsächlich hingeht. In Legacy-Organisationen ist der Effekt ausgeprägt, da Jahrzehnte angesammelter Prozesskontrollen, Übergaben und Freigaben jeweils eine Warteschlange hinzufügten, die seitdem niemand überprüft hat.

## How to Apply ◆

> Die Schritte, die die meiste Zeit in einem Legacy-Lieferprozess verbrauchen, sind üblicherweise die, an die niemand als Schritte denkt: das Warten auf die gemeinsam genutzte Testumgebung, der Change-Advisory-Board, das donnerstags tagt, das Release-Fenster alle drei Wochen.

- **Kartieren Sie ein echtes, kürzliches Element durchgängig**, statt den dokumentierten Prozess. Wählen Sie eine spezifische Änderung, die Produktion erreichte, und rekonstruieren Sie, was ihr geschah, mit Daten. Der dokumentierte Prozess und der tatsächliche Prozess weichen ab, und die Abweichung ist oft, wo die Verzögerung lebt.
- Beziehen Sie **die volle Spanne von Anfrage bis Produktion** ein, nicht nur den Entwicklungsteil. Der Großteil der Verschwendung sitzt vor Entwicklungsbeginn und nach ihrem Ende, sodass eine Karte, die bei "Ticket zugewiesen" beginnt und bei "Code gemergt" endet, nichts finden wird.
- Protokollieren Sie für jeden Schritt **Bearbeitungszeit und verstrichene Zeit getrennt**. Ein Code-Review mit fünfzehn Minuten Bearbeitungszeit und drei Tagen verstrichener Zeit ist ein Warteschlangenproblem, kein Review-Problem, und die zwei haben völlig unterschiedliche Fixes.
- **Protokollieren Sie die Übergaben** und wer auf jeder Seite ist. Jede Übergabe ist eine Warteschlange, ein Kontextverlust und eine potenzielle Nacharbeitsschleife. Sie zu zählen ist oft aufschlussreicher als die Zeiten selbst.
- Beachten Sie **Nacharbeitsschleifen**: wie oft Arbeit rückwärts geht, und warum. Arbeit, die durchschnittlich zweimal vom Testen zur Entwicklung zurückkehrt, ist ein Defektpräventionsproblem, das sich als Lieferproblem tarnt.
- **Führen Sie die Kartierung mit den Personen durch, die die Arbeit tun**, in einem Raum, an einer Wand. Die Karte ist nicht das Liefergut — die gemeinsame Erkenntnis ist es. Eine von einem Berater produzierte und dem Team präsentierte Karte überzeugt niemanden.
- **Greifen Sie zuerst die größte Wartezeit an**, nicht den nervigsten Schritt. Dies ist kontraintuitiv; die Schritte, über die sich Leute beschweren, sind üblicherweise kurz und irritierend, während die mehrtägigen Wartezeiten so normal sind, dass niemand sie erwähnt.
- Unterscheiden Sie **Wartezeiten, die etwas schützen, von Wartezeiten, die nichts schützen**. Ein Änderungsgenehmigungsausschuss, der in drei Jahren zwei Änderungen abgelehnt hat, ist eine Warteschlange ohne Ertrag; ein Sicherheitsreview, das echte Probleme erfasst, ist eine Warteschlange, die zu behalten und zu beschleunigen sich lohnt.
- **Kartieren Sie nach Änderungen erneut**, um zu verifizieren, dass die Verbesserung das Gesamtergebnis bewegt hat, nicht nur einen Schritt. Lokale Optimierungen schieben die Warteschlange häufig woanders hin, und nur die durchgängige Zahl zeigt, ob sich tatsächlich etwas verbessert hat.

## Tradeoffs ⇄

> Kartierung ist günstig und produziert häufig die einzelne wertvollste verfügbare Erkenntnis für eine Lieferorganisation, aber die identifizierten Verbesserungen liegen oft außerhalb der Autorität des Teams.

**Vorteile:**

- Wartezeit wird sichtbar, und da sie üblicherweise die Zykluszeit dominiert, sind hier die größten verfügbaren Verbesserungen.
- Verbesserungsaufwand wird durch Evidenz gelenkt, statt durch welchen Schritt die Personen, die sich am meisten beschweren, am irritierendsten finden.
- Übergaben und Genehmigungsschritte, die ihren Zweck überlebt haben, werden konkret identifiziert, mit den Kosten jedes einzelnen angehängt.
- Die Karte ist ein überzeugendes Artefakt für das Management, weil ein Diagramm, das drei Tage Arbeit und einunddreißig Tage Warten zeigt, ein Argument macht, das keine verbale Beschwerde kann.
- Sie schafft ein gemeinsames Verständnis über Rollen hinweg, die jeweils nur ihr eigenes Segment sehen, was häufig langjährige gegenseitige Schuldzuweisungen zwischen Entwicklung, Testing und Betrieb löst.

**Kosten und Risiken:**

- Der Workshop verbraucht mehrere Stunden Zeit vieler Personen und produziert keine Ausgabe, bis sich als Ergebnis etwas ändert.
- Die größten Wartezeiten gehören oft anderen Abteilungen — Änderungsausschüssen, Sicherheit, Beschaffung —, sodass das Team das Problem messen kann, ohne es beheben zu können.
- Ein einzelnes kartiertes Element ist möglicherweise nicht repräsentativ. Die Kartierung einer ungewöhnlich reibungslosen oder ungewöhnlich problematischen Änderung führt zu Schlussfolgerungen, die nicht verallgemeinern.
- Karten werden veraltet, während sich der Prozess ändert, und eine veraltete, für Entscheidungen genutzte Karte ist schlimmer als keine.
- Das Entfernen eines Kontrollschritts, der verschwenderisch erscheint, kann einen Schutz entfernen, dessen Wert unsichtbar war, gerade weil er funktionierte.

## How It Could Be

Ein Team, das eine Versicherungsschadensplattform unterstützte, glaubte, ihre Lieferung sei langsam, weil die Codebasis schwierig war. Sie kartierten eine kürzliche Änderung von der Anfrage bis zur Produktion: 4,5 Tage tatsächlicher Arbeit, verteilt über 47 Kalendertage. Die Karte zeigte 9 Tage Warten auf Geschäftsfreigabe der Anforderung, 6 Tage Warten auf eine gemeinsam genutzte Integrationsumgebung, 11 Tage Warten auf den zweiwöchentlichen Change-Advisory-Board, und 14 Tage Warten auf das monatliche Release-Fenster. Der Code selbst war nie die Einschränkung. In den nächsten zwei Quartalen adressierten sie die zwei größten Warteschlangen — Umstellung auf containerisierte Per-Branch-Umgebungen und Aushandlung eines schnellen Pfads durch den Änderungsausschuss für vorab genehmigte risikoarme Änderungstypen — und die mittlere Durchlaufzeit sank von 47 Tagen auf 16, ohne dass jemand die Codebasis anfasste.

Eine zweite Organisation nutzte die Karte, um eine Kontrolle zu verteidigen, statt sie zu entfernen. Die Kartierung zeigte, dass ihr Sicherheitsreview 4 Tage Wartezeit pro Änderung hinzufügte, und es gab Druck, es zu beseitigen. Dieselbe Übung protokollierte, dass das Review in 18 Monaten 14 echte Probleme erfasst hatte, von denen drei ernst waren. Statt den Schritt zu entfernen, verschoben sie ihn früher — Reviews von Designs statt fertigen Änderungen — und fügten automatisierte Prüfungen für die wiederkehrenden Kategorien hinzu. Die Wartezeit sank auf unter einen Tag, und die Erfassungsrate blieb gleich. Die Unterscheidung zwischen einer Warteschlange mit Ertrag und einer ohne war es, was die Diskussion produktiv statt positionell machte.
