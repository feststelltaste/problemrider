---
title: Kapazitätsbasierte Planung
description: Ableitung von Zusagen aus gemessenem historischem Durchsatz statt aus
  gewünschten Terminen, ausgedrückt als Bandbreiten mit angegebener Konfidenz.
category:
- Management
- Process
problems:
- unrealistic-deadlines
- unrealistic-schedule
- missed-deadlines
- planning-credibility-issues
- reduced-predictability
- constantly-shifting-deadlines
- large-estimates-for-small-changes
- deadline-pressure
- cascade-delays
- increased-time-to-market
- staff-availability-issues
- overworked-teams
- time-pressure
- delayed-project-timelines
- competing-priorities
- extended-cycle-times
- increased-stress-and-burnout
- increased-technical-shortcuts
- market-pressure
- mental-fatigue
- priority-thrashing
- team-demoralization
- uneven-work-flow
- uneven-workload-distribution
- budget-overruns
- changing-project-scope
- poor-planning
- project-resource-constraints
- scope-change-resistance
- stakeholder-confidence-loss
- stakeholder-dissatisfaction
- eager-to-please-stakeholders
- planning-dysfunction
- poor-project-control
- stakeholder-frustration
layout: solution
lang: de
en_slug: capacity-based-planning
related_solutions:
- slug: capacity-planning
  similarity: 0.75
- slug: short-iteration-cycles
  similarity: 0.7
- slug: work-in-progress-limits
  similarity: 0.7
- slug: iterative-development
  similarity: 0.7
- slug: delivery-performance-metrics
  similarity: 0.7
- slug: sustainable-pace-practices
  similarity: 0.65
---

## Description

Kapazitätsbasierte Planung leitet ab, wozu sich ein Team verpflichten kann, aus dem, was es tatsächlich geliefert hat, statt aus dem, was ein Termin von ihm verlangt. Sie ruht auf zwei Verschiebungen. Die erste ist Messung: Durchsatz und Zykluszeit werden über eine bedeutsame Historie hinweg aufgezeichnet und als Grundlage für Prognosen genutzt, was pro-Aufgabe-Schätzungen ersetzt, die zu einem Plan aggregiert werden. Die zweite ist Ehrlichkeit über Unsicherheit: Zusagen werden als Bandbreiten mit Konfidenzniveaus ausgedrückt statt als einzelne Termine, von denen jeder insgeheim weiß, dass sie optimistisch sind. In Legacy-Systemen ist dies wichtiger als anderswo, weil die dominante Quelle von Zeitplanabweichung nicht die geplante Arbeit ist — es ist die ungeplante Entdeckung, dass eine Änderung eine undokumentierte Abhängigkeit berührt, und keine Menge Vorabschätzung sagt das voraus. Historischer Durchsatz enthält diese Varianz jedoch bereits, weil sie auch in jeder vergangenen Periode vorhanden war.

## How to Apply ◆

> Legacy-Arbeit hat einen langen Schwanz an Überraschungen; eine Planungsmethode, die den Schwanz als Ausnahme behandelt, wird jedes Mal falsch liegen, während eine, die ihn als normale Systemeigenschaft misst, es nicht wird.

- Beginnen Sie mit der **Messung des tatsächlichen Durchsatzes** für mindestens acht bis zwölf abgeschlossene Perioden: wie viele Elemente welchen Typs fertiggestellt wurden und wie lange jedes von Start bis Fertigstellung dauerte. Nutzen Sie die Einheiten, in denen das Team bereits arbeitet. Die absoluten Zahlen zählen weit weniger als ihre Streuung.
- Prognostizieren Sie mit **aus dieser Historie abgeleiteten Bandbreiten**, nicht mit einer einzelnen Zahl. Formulieren Sie Pläne als „fünfundachtzig Prozent zuversichtlich bis Ende März, fünfzig Prozent zuversichtlich bis Mitte Februar." Die Streuung vergangener Leistung ist das ehrliche Maß der Unsicherheit, und sie zu formulieren verwandelt einen Streit über Optimismus in eine Diskussion über akzeptables Risiko.
- Ziehen Sie **bekannte Nicht-Projektzeit vor der Zusage ab**, nicht danach. Support-Rotationen, Vorfallreaktion, Meetings, Feiertage, Onboarding und die Unterbrechungslast eines Legacy-Systems sind kein Overhead, der heroisch absorbiert werden soll — sie sind Kapazität, die nicht existiert. Teams, die mit hundert Prozent der nominalen Kapazität planen, verpassen Termine strukturell, nicht gelegentlich.
- Verfolgen und veröffentlichen Sie die **Unterbrechungslast** als separate Zahl. In vielen Legacy-Teams macht sie dreißig bis fünfzig Prozent der Kapazität aus, und sie ist üblicherweise in der Planung unsichtbar, weil sie im Plan unsichtbar ist. Sie zu einer Zahl zu machen ändert sowohl die Prognose als auch schließlich die Investitionsentscheidungen, die sie verringern würden.
- Wenn ein Termin extern fixiert ist, **variieren Sie den Umfang statt der Prognose**. Berechnen Sie, was in die verfügbare Kapazität bei einer angegebenen Konfidenz passt, und präsentieren Sie das als Liefergegenstand. Sich zu einem Umfang zu verpflichten, den die Kapazität nicht trägt, schafft keine Kapazität; es verschiebt die Entdeckung des Defizits auf den Punkt, an dem die wenigsten Optionen verbleiben.
- Nutzen Sie **Referenzklassen-Vergleich** für große oder unvertraute Arbeit: Finden Sie die drei ähnlichsten Dinge, die das Team abgeschlossen hat, und wie lange sie tatsächlich gedauert haben. Dies ist ein weit besserer Prädiktor als Zerlegung-und-Schätzung, was systematisch die Arbeit auslässt, an die niemand gedacht hat — was in Legacy-Systemen das meiste davon ist.
- **Prognostizieren Sie in festem Rhythmus neu**, unter Nutzung aktualisierter Ist-Werte, und behandeln Sie eine sich bewegende Prognose als Information statt als Versagen. Ein Plan, der sich in einer Legacy-Umgebung nie ändert, ist ein Plan, der aufgehört hat, die Realität zu verfolgen.
- Erfassen Sie **Schätzung versus Ist-Wert** für eine Stichprobe der Arbeit und überprüfen Sie sie vierteljährlich. Der Zweck ist Kalibrierung, nicht Verantwortlichkeit; wenn die Überprüfung genutzt wird, um Einzelpersonen zu kritisieren, werden sich die Schätzungen anpassen, um ihre Autoren zu schützen, und die Daten werden wertlos.
- Präsentieren Sie Prognosen an Stakeholder mit **angehängter Evidenz** — der Durchsatzhistorie, der Konfidenzgrundlage, der angenommenen Unterbrechungslast. Eine Prognose, die als nackter Termin ankommt, lädt zu Verhandlung ein; eine, die mit ihrer Herleitung ankommt, lädt zu einer Diskussion darüber ein, welche Annahme geändert werden sollte.

## Tradeoffs ⇄

> Kapazitätsbasierte Planung produziert Prognosen, die erheblich genauer und erheblich weniger willkommen sind, weil sie den Optimismus entfernt, der frühere Pläne akzeptabel machte.

**Vorteile:**

- Die Prognosegenauigkeit verbessert sich erheblich, weil historischer Durchsatz bereits die Unterbrechungen, Nacharbeit und Überraschungen enthält, die pro-Aufgabe-Schätzungen systematisch ausschließen.
- Planungsglaubwürdigkeit erholt sich über die Zeit, da eingehaltene Termine schneller Vertrauen aufbauen als ambitionierte Termine, die verpasst werden.
- Die echten Kosten der Unterbrechungslast werden sichtbar und quantifiziert, was üblicherweise die Voraussetzung für jede Investition zu ihrer Verringerung ist.
- Der Termindruck sinkt, weil Zusagen aus Evidenz abgeleitet statt nach unten verhandelt werden, was die strukturelle Überzusage entfernt, die anhaltende Überstunden antreibt.
- Umfangsgespräche finden früh statt, wenn Umfangsanpassung günstig ist, statt in den letzten Wochen, wenn sie es nicht ist.

**Kosten und Risiken:**

- Die ersten ehrlichen Prognosen sind üblicherweise weit später als das, was der Organisation gesagt wurde, und diese Nachricht zu überbringen ist politisch kostspielig, unabhängig davon, wie gut die Evidenz präsentiert wird.
- Bandbreiten und Konfidenzniveaus sind vielen Stakeholdern unvertraut, die „fünfundachtzig Prozent zuversichtlich bis März" hören und „März" notieren könnten. Konsistente, geduldige Kommunikation ist erforderlich, und es braucht mehrere Zyklen.
- Die Methode braucht eine bedeutsame Historie, sodass neu gebildete Teams oder Teams, die in ein unvertrautes Subsystem eintreten, anfänglich wenig zum Arbeiten haben.
- Historischer Durchsatz setzt Kontinuität voraus. Eine bedeutsame Änderung in Teamzusammensetzung, Technologie oder der Natur der Arbeit invalidiert die Baseline, und Legacy-Modernisierungsbemühungen beinhalten oft genau solche Änderungen.
- Messung kann korrumpiert werden, wenn Durchsatz zu einem Ziel wird. Teams optimieren, was gezählt wird, typischerweise durch feinere Aufteilung der Arbeit, was die Metrik aufbläht, ohne die Lieferung zu verbessern.

## How It Could Be

Ein Team, das ein Telekommunikations-Abrechnungssystem pflegte, hatte elf seiner letzten zwölf vierteljährlichen Zusagen verpasst, und Planung war zu einem Ritual degeneriert, bei dem jeder insgeheim die genannten Termine verdoppelte. Sie verbrachten zwei Wochen damit, tatsächlichen Durchsatz aus ihrem Ticket-System zu rekonstruieren, und fanden, dass sie über das vorherige Jahr zwischen vierzehn und sechsundzwanzig Elemente pro Monat abgeschlossen hatten, wobei Support-Arbeit einen ungemessenen, aber substanziellen Anteil verbrauchte. Ihre nächste Zusage wurde als Bandbreite mit Konfidenzniveaus präsentiert, und die Support-Last wurde separat auf ungefähr achtunddreißig Prozent der Kapazität quantifiziert. Die unmittelbare Reaktion ihres Direktors war, dass die Prognose inakzeptabel sei. Das nachfolgende Gespräch drehte sich jedoch um die achtunddreißig Prozent — von denen niemand gewusst hatte — und führte zur Finanzierung eines dedizierten Support-Ingenieurs. Das Team erfüllte seine nächsten drei vierteljährlichen Zusagen.

Ein zweites Team sah sich einer fixen regulatorischen Frist neun Monate im Voraus gegenüber und wurde gefragt, ob der volle Compliance-Umfang passen würde. Statt unter Druck mit Ja zu antworten, nutzten sie Referenzklassen-Vergleich gegen zwei vorherige Compliance-Bemühungen und prognostizierten, dass ungefähr siebzig Prozent des Umfangs bei fünfundachtzig Prozent Konfidenz passen. Sie präsentierten die siebzig Prozent als zugesagten Liefergegenstand und die verbleibenden dreißig Prozent als priorisierte Elemente, die nur landen würden, wenn die frühere Arbeit ungewöhnlich gut lief. Zwei der verschobenen Elemente stellten sich bei genauerer Lektüre der Regulierung als unnötig heraus, und der verpflichtende Umfang wurde sechs Wochen vor der Frist geliefert — das erste Mal in der Erinnerung dieser Organisation, dass ein Compliance-Projekt am Ende keinen Krisen-Endspurt erforderte.
