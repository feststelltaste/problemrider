---
title: Regelmäßige Stakeholder-Demonstrationen
description: Arbeitende Software den Personen zeigen, die sie angefragt
  haben, in festem Takt, sodass Missverständnisse in Tagen statt erst bei
  der Auslieferung sichtbar werden.
category:
- Communication
- Business
- Process
problems:
- stakeholder-confidence-loss
- stakeholder-dissatisfaction
- stakeholder-frustration
- feedback-isolation
- eager-to-please-stakeholders
- feature-factory
- product-direction-chaos
- planning-credibility-issues
- reduced-feature-quality
- inadequate-requirements-gathering
- requirements-ambiguity
- delayed-value-delivery
- missed-deadlines
- cascade-delays
- changing-project-scope
- constantly-shifting-deadlines
- deadline-pressure
- delayed-project-timelines
- gold-plating
- incomplete-projects
- poor-communication
- poor-planning
- scope-change-resistance
- unrealistic-deadlines
- unrealistic-schedule
- communication-risk-outside-project
- declining-business-metrics
- feature-creep
- frequent-changes-to-requirements
- market-pressure
- poor-project-control
- unclear-goals-and-priorities
- unproductive-meetings
- process-software-misfit
layout: solution
lang: de
en_slug: regular-stakeholder-demonstrations
related_solutions:
- slug: continuous-feedback
  similarity: 0.8
- slug: stakeholder-feedback-loops
  similarity: 0.8
- slug: on-site-customer
  similarity: 0.65
- slug: short-iteration-cycles
  similarity: 0.65
- slug: iterative-development
  similarity: 0.65
- slug: prototyping
  similarity: 0.65
---

## Description

Eine regelmäßige Demonstration ist eine kurze Sitzung mit festem Takt, in der das Team arbeitende Software zeigt — keine Folien, keinen Status, keine Prozentzahlen des Fertigstellungsgrads — den Personen, die sie angefragt haben und die sie nutzen werden. Ihre Funktion ist es, die abstrakte Frage „ist das, was Sie meinten" in eine konkrete zu verwandeln, die durch Hinsehen beantwortet werden kann. Schriftliche Anforderungen sind immer unvollständig, und die Lücke zwischen dem, was ein Stakeholder angefragt hat, und dem, was er meinte, ist nicht durch sorgfältigeres Nachfragen entdeckbar; sie ist entdeckbar, indem man ihnen etwas zeigt. Der Takt zählt ebenso wie der Inhalt, weil eine Demonstration alle zwei Wochen das mögliche Missverständnis auf zwei Wochen Arbeit begrenzt. In der Legacy-Modernisierung dient die Praxis einem zweiten Zweck: Sie ist der einzige verlässliche Weg, Fortschritt bei Arbeit zu zeigen, deren sichtbare Ausgabe sonst monatelang null ist.

## How to Apply ◆

> Legacy-Arbeit produziert oft über lange Strecken nichts, was ein Stakeholder sehen kann, was genau die Bedingung ist, unter der Vertrauen erodiert und Druck aufgebaut wird.

- Demonstrieren Sie **laufende Software in einer Umgebung, die Produktion ähnelt**, mit realistischen Daten. Ein Durchgang durch ein Design oder eine Beschreibung dessen, was gebaut wurde, legt nicht die Missverständnisse offen, die das Sehen des tatsächlichen Verhaltens offenlegt.
- Halten Sie sie nach einem **festen Zeitplan** ab, ob viel zu zeigen ist oder nicht. Sie abzusagen, weil der Fortschritt dünn war, entfernt das Feedback genau dann, wenn das Team am meisten prüfen muss, dass es noch auf Kurs ist, und es lehrt Stakeholder, dass das Meeting nur gute Nachrichten signalisiert.
- Laden Sie **die Personen ein, die das System tatsächlich nutzen werden**, nicht nur ihre Manager. Die Lücke zwischen dem, was ein Abteilungsleiter beschreibt, und dem, was die Person, die die Arbeit macht, braucht, ist der Ursprung eines großen Anteils unbrauchbarer Features.
- Lassen Sie die **Person, die es gebaut hat, es demonstrieren**. Fragen werden direkt beantwortet, und der Entwickler hört die Reaktion ungefiltert, was nachfolgende Entscheidungen mehr verändert als jede weitergegebene Zusammenfassung.
- **Zeigen Sie das Unfertige und Unvollkommene** bewusst. Eine Demonstration, die nur polierte Arbeit zeigt, verzögert Feedback, bis die Kursänderung teuer ist, und trainiert Stakeholder darauf, ein fertiges Erscheinungsbild zu erwarten, was dann einschränkt, was das Team bereit ist zu zeigen.
- Demonstrieren Sie für Arbeit mit **keiner sichtbaren Oberfläche** — eine Migration, eine Performance-Anstrengung, ein Abhängigkeits-Upgrade — stattdessen die Evidenz: die Vorher-Nachher-Messung, den Parallelbetrieb-Vergleich, den nun vom neuen Pfad bedienten Traffic. Der Punkt ist, dass etwas Verifizierbares gezeigt wird, nicht dass es ein Bildschirm ist.
- **Zeichnen Sie Entscheidungen und angefragte Änderungen** während der Sitzung auf und speisen Sie sie in das priorisierte Backlog ein statt in die aktuelle Arbeit. Eine Demonstration, die still den Umfang erweitert, ist, wie ein wohlmeinendes Team unfähig wird, irgendetwas fertigzustellen.
- Halten Sie sie **kurz und unprobt**. Eine Sitzung, deren Vorbereitung zwei Tage braucht, wird seltener vorbereitet, und Vorbereitungsaufwand tendiert dazu, Dinge fertig aussehen zu lassen statt sie tatsächlich betrachten zu lassen.
- Nutzen Sie den Takt, um **die Glaubwürdigkeit aufzubauen, die Planungsdiskussionen brauchen**. Stakeholder, die sechs Monate lang alle zwei Wochen arbeitende Software gesehen haben, reagieren sehr anders auf eine Prognose als solche, die Statusberichte gesehen haben.

## Tradeoffs ⇄

> Häufige Demonstrationen erfassen Missverständnisse früh und bauen Vertrauen wieder auf, verbrauchen aber Zeit beschäftigter Personen und legen Arbeit offen, bevor es angenehm ist, sie zu zeigen.

**Vorteile:**

- Missverständnisse werden innerhalb einer Taktperiode erfasst statt bei der Auslieferung, was den Unterschied zwischen einer Anpassung und einer Neuschreibung ausmacht.
- Stakeholder-Vertrauen erholt sich, weil Fortschritt beobachtet statt behauptet wird — dies ist üblicherweise weit effektiver als jede Verbesserung der Berichterstattung.
- Das Team hört Reaktionen direkt, was nachfolgende Designentscheidungen auf Weisen verbessert, die schriftliches Feedback nicht erreicht.
- Anfragen werden an einem Ort zu vorhersagbarer Zeit erfasst, was die ad hoc Seitenkanal-Anfragen reduziert, die sonst kontinuierlich eintreffen und Planung stören.
- Unsichtbare Arbeit wird sichtbar, wenn die Evidenz demonstriert wird, was Modernisierungsanstrengungen davor schützt, wegen scheinbaren Fortschrittsmangels abgesagt zu werden.

**Kosten und Risiken:**

- Sie verbraucht die Zeit leitender Stakeholder auf wiederkehrender Basis, und die Teilnahme sinkt, wenn die Sitzungen nicht durchgängig den Besuch wert sind.
- Demonstrationen laden Anfragen ein, und ohne die Disziplin, sie an das Backlog weiterzuleiten, werden sie zu einem Mechanismus kontinuierlicher Umfangserweiterung.
- Die Vorbereitung eines demonstrierbaren Zustands kann Prioritäten in Richtung dessen verzerren, was gut aussieht, statt dessen, was zählt, besonders wenn das Publikum hauptsächlich auf die Oberflächenerscheinung reagiert.
- Unfertige Arbeit zu zeigen erfordert genug Vertrauen, dass Unfertigkeit nicht als Inkompetenz gelesen wird, und in einer Beziehung mit geringem Vertrauen können die frühen Sitzungen die Dinge verschlimmern, bevor sie besser werden.
- Manche Legacy-Arbeit hat echt wochenlang nichts zu zeigen, und eine Demonstration in diesen Perioden zu erzwingen produziert konstruierten Inhalt, der die Glaubwürdigkeit des Formats untergräbt.

## How It Could Be

Ein Team, das die Preis-Engine einer E-Commerce-Plattform neu baute, arbeitete fünf Monate lang gegen eine schriftliche Spezifikation und demonstrierte das Ergebnis am Ende. Ungefähr vierzig Prozent davon mussten überarbeitet werden: Die Beschreibung des gestapelten Rabattverhaltens in der Spezifikation entsprach dem Dokument, das ihr Autor geschrieben hatte, aber nicht, wie die Preisanalysten tatsächlich arbeiteten, was manuelle Übersteuerungen beinhaltete, die nirgendwo aufgeschrieben worden waren. Beim nächsten größeren Aufwand demonstrierte das Team alle zwei Wochen vor zwei Analysten und einem Manager. Das äquivalente Missverständnis — diesmal darüber, wie Steuerbefreiungen mit Mengenrabatten interagierten — tauchte in der dritten Sitzung auf, elf Tage in die Arbeit hinein, und kostete einen Nachmittag zur Korrektur.

Ein Modernisierungsteam stand vor dem umgekehrten Problem: acht Monate einer Datenbankmigration mit nichts Sichtbarem zu zeigen und wachsendem Druck, die Investition zu rechtfertigen. Sie begannen, Evidenz statt Bildschirme zu demonstrieren. Jede Sitzung zeigte das Parallelbetrieb-Vergleichs-Dashboard: wie viele Datensatztypen nun in beide Systeme geschrieben wurden, wie viele Diskrepanzen diese Woche gefunden worden waren und welche gelöst worden waren. Die Fragen der Steuerungsgruppe verschoben sich von „wann wird das fertig sein" zu substanzieller Diskussion der Diskrepanzkategorien, und der Aufwand wurde zweimal ohne Widerspruch verlängert. Zwei der in diesen Sitzungen gezeigten Diskrepanzmuster stellten sich als bereits bestehende Datenqualitätsprobleme heraus, die die Geschäftsseite jahrelang manuell umgangen hatte und die niemand zuvor hatte quantifizieren können.
