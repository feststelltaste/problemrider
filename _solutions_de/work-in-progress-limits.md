---
title: Work-in-Progress-Limits
description: Begrenzung, wie viele Elemente das Team begonnen, aber
  nicht abgeschlossen haben darf, sodass Arbeit fertiggestellt statt
  angehäuft wird.
category:
- Process
- Team
- Management
problems:
- context-switching-overhead
- priority-thrashing
- uneven-work-flow
- uneven-workload-distribution
- work-blocking
- extended-cycle-times
- delayed-project-timelines
- constant-firefighting
- maintenance-bottlenecks
- reduced-team-productivity
- mental-fatigue
- cascade-delays
- incomplete-projects
- delayed-issue-resolution
- avoidance-behaviors
- competing-priorities
- developer-frustration-and-burnout
- extended-review-cycles
- increased-stress-and-burnout
- increased-time-to-market
- procrastination-on-complex-tasks
- reduced-individual-productivity
- reduced-predictability
- reduced-review-participation
- resource-waste
- review-bottlenecks
- review-process-breakdown
- rushed-approvals
- team-demoralization
- team-members-not-engaged-in-review-process
- code-review-inefficiency
- constantly-shifting-deadlines
- deadline-pressure
- insufficient-code-review
- long-lived-feature-branches
- missed-deadlines
- overworked-teams
- reduced-code-submission-frequency
- staff-availability-issues
- superficial-code-reviews
- time-pressure
- unrealistic-schedule
- author-frustration
- bottleneck-formation
- inadequate-code-reviews
- perfectionist-review-culture
layout: solution
lang: de
en_slug: work-in-progress-limits
related_solutions:
- slug: capacity-based-planning
  similarity: 0.7
- slug: sustainable-pace-practices
  similarity: 0.7
- slug: value-stream-mapping
  similarity: 0.7
- slug: delivery-performance-metrics
  similarity: 0.7
- slug: definition-of-ready
  similarity: 0.65
- slug: team-retrospectives
  similarity: 0.65
---

## Description

Ein Work-in-Progress-Limit ist eine vereinbarte maximale Anzahl von Elementen, die gleichzeitig in einem unfertigen Zustand sein dürfen — pro Person, pro Workflow-Stufe oder pro Team. Es ist die direkteste verfügbare Intervention gegen das Muster, bei dem alles begonnen, nichts fertiggestellt und jeder beschäftigt ist. Der Mechanismus ist bewusst unbequem: Wenn das Limit erreicht ist, darf keine neue Arbeit aufgenommen werden, sodass das Team entweder etwas fertigstellen oder lösen muss, was auch immer es blockiert. Diese erzwungene Konfrontation mit Blockern ist der eigentliche Wert; das Limit selbst ist nur der Auslöser. In der Legacy-Wartung ist der Effekt ausgeprägt, weil unfertige Arbeit in einem brüchigen System nicht nur untätig ist — halb migrierte Datenstrukturen, teilweise angewendete Refaktorierungen und aufgegebene Branches erhöhen aktiv das Risiko und die Kosten von allem anderen, das das Team anfasst.

## How to Apply ◆

> Legacy-Teams werden gleichzeitig in viele Richtungen gezogen — Produktionsvorfälle, Migrationsarbeit, Feature-Anfragen und Support-Eskalationen —, sodass Limits unterbrechungsgetriebene Arbeit berücksichtigen müssen, statt so zu tun, als existiere sie nicht.

- Machen Sie aktuelle Arbeit in Bearbeitung **sichtbar, bevor Sie sie begrenzen**. Setzen Sie jedes begonnene, aber unfertige Element auf ein Board, einschließlich der unsichtbaren: offene Branches, wartende Reviews, halbfertige Untersuchungen und Support-Tickets, die jemand still trägt. Teams sind routinemäßig schockiert von der Anzahl, und die Sichtbarkeit allein ändert Verhalten, bevor irgendein Limit gesetzt wird.
- Setzen Sie das erste Limit **knapp unter dem aktuellen Durchschnitt**, nicht bei einer theoretisch idealen Zahl. Wenn das Team derzeit achtzehn Elemente trägt, beginnen Sie bei vierzehn. Ein Limit, das sofort von der Realität verletzt wird, wird innerhalb einer Woche verworfen; ein Limit, das sanft beißt, bleibt bestehen.
- Wenden Sie das Limit **pro Workflow-Stufe an, nicht nur insgesamt**, sodass Engpässe sichtbar werden. Eine Obergrenze für "in Review" ist üblicherweise das wertvollste erste Limit für Teams mit Review-Engpässen, weil es Reviewen zwingt, mit dem Beginnen neuer Arbeit zu konkurrieren, statt immer dagegen zu verlieren.
- Definieren Sie die **Stop-the-Line-Regel explizit**: Wenn eine Stufe an ihrem Limit ist, helfen Personen, die neue Arbeit aufgenommen hätten, stattdessen, bestehende Arbeit zu beenden oder zu entblockieren. Ohne diese Regel schafft das Limit nur Leerlaufzeit und wird als verschwenderisch aufgegeben.
- Reservieren Sie **explizite Kapazität für Unterbrechungsarbeit**, statt sie das Limit brechen zu lassen. Ein dedizierter Slot — eine Person in Support-Rotation, oder zwei reservierte WIP-Slots für Vorfälle — verhindert, dass ungeplante Arbeit das Limit still ins Unbegrenzte erweitert, und macht die echten Kosten der Unterbrechungslast in der Planung sichtbar.
- Verfolgen Sie **blockierte Elemente separat** und überprüfen Sie sie täglich. Der Wert eines WIP-Limits zeigt sich nur, wenn Blocker eskaliert werden; wenn blockierte Elemente einfach innerhalb des Limits sitzen, ist das Team gedeckelt, aber fließt nicht. Ein Blocker, der älter als einen Tag ist, sollte einen benannten Verantwortlichen und einen Eskalationspfad haben.
- Messen Sie **Zykluszeit und Abschlussrate**, nicht Auslastung. Das erwartete Ergebnis ist, dass Personen individuell weniger beschäftigt sind und das Team mehr fertigstellt, was wie eine Regression bei jeder Auslastungsmetrik aussieht und dem Management im Voraus erklärt werden muss.
- Überprüfen Sie das Limit alle paar Wochen und **senken Sie es, während der Fluss sich weiter verbessert**. Das Limit ist ein Abstimmungsparameter, keine Richtlinie; wenn das Senken aufhört, die Zykluszeit zu verbessern, oder echten Leerlauf verursacht, war der vorherige Wert richtig.

## Tradeoffs ⇄

> Die Begrenzung von Arbeit in Bearbeitung verbessert Durchsatz und Vorhersagbarkeit, erfordert aber, sichtbaren Leerlauf zu akzeptieren und Nein zu Arbeit zu sagen, die bereits zugesagt wurde.

**Vorteile:**

- Die Zykluszeit sinkt, oft dramatisch, weil Elemente aufhören, sich hintereinander zu stauen, während sie nominell in Bearbeitung sind. Dies ist Arithmetik statt Motivation: weniger gleichzeitige Arbeit bedeutet weniger Wartezeit pro Element.
- Kontextwechsel sinken, was eine große Menge effektiver Kapazität in Legacy-Arbeit zurückgewinnt, wo das Neuladen des Kontexts eines komplexen Moduls teuer ist.
- Blocker treten sofort zutage und werden eskaliert, statt still absorbiert zu werden, indem etwas anderes begonnen wird.
- Engpässe werden sichtbar und lokalisierbar. Eine Stufe, die konstant an ihrem Limit ist, identifiziert genau, wo die Kapazitätseinschränkung des Teams liegt — üblicherweise Review oder Testing.
- Halbfertige Arbeit in der Codebasis nimmt ab, was das Risiko reduziert, dass teilweise angewendete Änderungen schlecht miteinander interagieren.

**Kosten und Risiken:**

- Das Limit ist nur so stark wie die Bereitschaft des Managements, es zu respektieren. Wenn neue Arbeit trotzdem hineingedrückt wird, bleibt dem Team die Zeremonie ohne den Nutzen, und das Vertrauen in die Praxis wird verbraucht.
- Sichtbarer Leerlauf ist politisch schwierig. Ein Ingenieur ohne etwas zum Aufnehmen wirkt wie Verschwendung für einen Beobachter, der Geschäftigkeit misst, und dies muss aktiv verteidigt werden.
- Schlecht gewählte Limits verursachen echte Stauungen, besonders in kleinen Teams, wo ein blockiertes Element einen großen Anteil der Obergrenze verbrauchen kann.
- Unterbrechungslastige Umgebungen können Limits bedeutungslos machen, es sei denn, Unterbrechungskapazität wird explizit reserviert, und sie zu reservieren bedeutet zuzugeben, wie viel Kapazität die Unterbrechungslast tatsächlich verbraucht.
- Die Praxis legt unangenehme Fakten offen — dass das Team an einer bestimmten Stufe der Engpass ist, oder dass ein abhängiges Team nie antwortet —, die manche Organisationen lieber nicht dokumentiert hätten.

## How It Could Be

Ein fünfköpfiges Team, das ein Krankenhausterminierungssystem pflegte, verfolgte zum ersten Mal seine laufende Arbeit und zählte dreiundzwanzig begonnene Elemente: sechs über einen Monat alte Branches, fünf auf jemanden wartende Reviews, vier Untersuchungen ohne klaren nächsten Schritt und acht Tickets in aktiver Entwicklung. Sie setzten ein Team-Limit von zehn und ein Review-Stufen-Limit von drei. In der ersten Woche begann fast nichts Neues; das Team verbrachte vier Tage damit, veraltete Branches zu schließen, von denen zwei gänzlich aufgegeben wurden und einer sich als Konflikt mit Arbeit herausstellte, die jemand anderes gerade abgeschlossen hatte. Im folgenden Quartal sank die mittlere Zykluszeit von neunzehn Tagen auf sechs, und die Anzahl der pro Monat abgeschlossenen Elemente stieg um ungefähr vierzig Prozent, trotz keiner Änderung der Teamgröße.

Ein Plattformteam, das unter konstantem Feuerwehrlöschen litt, nutzte eine andere Variante: Zwei ihrer acht WIP-Slots wurden dauerhaft für Vorfälle und dringenden Support reserviert, und geplante Arbeit durfte nie sechs überschreiten. Die Reservierung machte die Unterbrechungslast zum ersten Mal messbar — sie verbrauchte konsequent beide Slots und verlangte oft einen dritten. Diese Daten, präsentiert als "ein Viertel unserer Kapazität geht in diesem Subsystem in ungeplante Arbeit", waren es, was schließlich eine dedizierte Stabilisierungsanstrengung an den zwei Modulen rechtfertigte, die die meisten Vorfälle generierten. Sechs Monate später passte die Unterbrechungslast bequem in einen Slot, und die freigesetzte Kapazität ging zurück zu geplanter Arbeit.
