---
title: Verbesserungsbudget
description: Reservierung eines festen, geschützten Anteils der Kapazität jeder
  Periode für Wartung, Refactoring und Risikoreduktion, sodass Verbesserung nie
  mit Features um Genehmigung konkurriert.
category:
- Management
- Process
- Code
problems:
- maintenance-paralysis
- increasing-brittleness
- increased-technical-shortcuts
- feature-creep-without-refactoring
- system-stagnation
- delayed-bug-fixes
- partial-bug-fixes
- maintenance-cost-increase
- short-term-focus
- time-pressure
- reduced-innovation
- inability-to-innovate
- high-technical-debt
- accumulation-of-workarounds
- competing-priorities
- developer-frustration-and-burnout
- high-turnover
- increased-stress-and-burnout
- reduced-individual-productivity
- team-demoralization
- tool-limitations
- unmotivated-employees
- deadline-pressure
- maintenance-bottlenecks
- overworked-teams
- reduced-team-productivity
- refactoring-avoidance
- test-debt
- brittle-codebase
layout: solution
lang: de
en_slug: improvement-budget
related_solutions:
- slug: technical-debt-backlog
  similarity: 0.7
- slug: total-cost-of-ownership-transparency
  similarity: 0.7
- slug: performance-budgets
  similarity: 0.7
- slug: incremental-refactoring
  similarity: 0.7
- slug: code-metrics
  similarity: 0.65
- slug: sustainable-pace-practices
  similarity: 0.65
---

## Description

Ein Verbesserungsbudget ist ein fester Anteil der Kapazität jeder Periode — üblicherweise zehn bis zwanzig Prozent —, reserviert für Arbeit, die das System verbessert statt es zu erweitern: Refactoring, Testabdeckung, Abhängigkeits-Upgrades, Tooling, Dokumentation und Entfernung toten Codes. Seine prägende Eigenschaft ist, dass das Budget einmal, auf Ebene der Richtlinie, zugewiesen wird, statt Posten für Posten gerechtfertigt zu werden. Dies ist der ganze Sinn. Verbesserungsarbeit verliert jeden einzelnen Vergleich gegen ein Feature, weil ihr Nutzen diffus und verzögert ist, während der des Features spezifisch und unmittelbar ist. Ein Team, das dieses Argument jedes Mal gewinnen muss, wird es jedes Mal verlieren, weshalb Systeme verkommen, selbst wenn alle Beteiligten zustimmen, dass Wartung wichtig ist. Die Entscheidung von der Posten-Ebene auf die Kapazitäts-Ebene zu verschieben ist es, was Verbesserung strukturell möglich macht.

## How to Apply ◆

> Legacy-Systeme sammeln Verkommen schneller an, als Teams es opportunistisch angehen können, und die Module, die am meisten Verbesserung brauchen, sind meist diejenigen, die niemand ohne geschützte Zeit anfassen möchte.

- Vereinbaren Sie den **Anteil explizit mit demjenigen, der die Kapazität des Teams kontrolliert**, und erfassen Sie ihn dort, wo später darauf verwiesen werden kann. Zehn Prozent reichen, um Verkommen in einem stabilen System zu stoppen; ein bereits in Schwierigkeiten steckendes System braucht typischerweise zwanzig Prozent oder mehr über einen anhaltenden Zeitraum, bevor sich der Trend umkehrt.
- Machen Sie das Budget zu **Kapazität, nicht Kalenderzeit**. „Jeden Freitag" scheitert am ersten arbeitsreichen Freitag, und dann an jedem darauffolgenden. Ein reservierter Anteil der Kapazität jeder Iteration übersteht Druck besser, weil das Wegnehmen davon eine explizite Entscheidung erfordert statt eines stillen Standardfalls.
- **Lassen Sie das Team entscheiden, wofür das Budget ausgegeben wird**, innerhalb eines festgelegten Umfangs. Das Team weiß, welche Module es am meisten kosten, und eine Genehmigung für jeden Posten zu verlangen führt genau den Posten-für-Posten-Wettbewerb wieder ein, den das Budget beseitigen soll.
- Priorisieren Sie das Budget mit **Evidenz statt Präferenz**: Änderungshäufigkeit gekreuzt mit Defektdichte identifiziert die Bereiche, in denen Verbesserung am meisten zurückgibt. Das Refactoring eines Moduls, das niemand seit vier Jahren angefasst hat, ist Aufwand, der nichts zurückgibt, wie unangenehm dieses Modul auch zu lesen sein mag.
- Verlangen Sie dieselbe **Sichtbarkeit wie Feature-Arbeit** — die Posten erscheinen auf demselben Board, im selben Review, mit derselben Definition von „erledigt". Unsichtbare Verbesserungsarbeit ist nicht von keiner Verbesserungsarbeit zu unterscheiden, wenn ihr Wert sechs Monate später infrage gestellt wird.
- Erfassen Sie das **Ergebnis dessen, was das Budget gekauft hat** in konkreten Begriffen: Build-Zeit von elf auf vier Minuten reduziert, diese Klasse von Produktionsdefekt beseitigt, diese Abhängigkeit jetzt wieder unterstützt. Verbesserungsbudgets werden gekürzt, wenn ihre Effekte nicht benannt werden können, und die Effekte sind fast immer benennbar, wenn sie jemand zum Zeitpunkt aufschreibt.
- Definieren Sie im Voraus die **Bedingungen, unter denen das Budget ausgesetzt werden darf** — ein echter Produktionsnotfall, eine harte regulatorische Frist — und verlangen Sie, dass ausgesetzte Kapazität zurückgezahlt statt erlassen wird. Ohne eine Rückzahlungsregel wird Aussetzung durch Anhäufung permanent.
- Kombinieren Sie das Budget mit der **Pfadfinderregel** für opportunistische Verbesserung: Kleine Aufräumarbeiten innerhalb des Bereichs, den eine Änderung ohnehin anfasst, sind Teil normaler Arbeit und werden nicht dem Budget belastet. Das Budget finanziert die Verbesserungen, die zu groß sind, um beiläufig zu geschehen.
- Überprüfen Sie den Anteil **vierteljährlich gegen den Trend**, nicht die Stimmung. Wenn Defektraten, Zykluszeiten und Vorfallhäufigkeit sich immer noch verschlechtern, ist das Budget zu klein, um zu zählen, und sollte ehrlich erhöht oder eingestellt werden statt als Geste aufrechterhalten zu werden.

## Tradeoffs ⇄

> Ein geschütztes Budget ist der einzige zuverlässige Weg, Verbesserung in einer feature-getriebenen Organisation zu finanzieren, aber es verpflichtet Kapazität im Voraus für Arbeit, deren Erträge echt, verzögert und schwer zuzuordnen sind.

**Vorteile:**

- Verbesserungsarbeit passiert tatsächlich, statt ewig auf „nach dem aktuellen Termin" verschoben zu werden — ein Zustand, den kein Legacy-System je erreicht hat.
- Das Team muss Wartung nicht mehr einzeln rechtfertigen, was ein wiederkehrendes und demoralisierendes Argument und erheblichen Managementaufwand beseitigt.
- Verkommen verlangsamt sich messbar. Systeme mit einem anhaltenden Verbesserungsbudget zeigen flacheres Wachstum bei Defektraten, Build-Zeiten und Änderungskosten als vergleichbare Systeme ohne eines.
- Entwickler gewinnen etwas Kontrolle über die Umgebung zurück, in der sie arbeiten, was einer der stärksten Prädiktoren für Bindung unter den Pflegern schwieriger Systeme ist.
- Die allmähliche Anhäufung von Abkürzungen unter Termindruck wird sichtbar, weil das Budget einen offensichtlichen Ort bietet, sie zurückzuzahlen, und ihre Abwesenheit davon auffällig wird.

**Kosten und Risiken:**

- Zehn bis zwanzig Prozent der Kapazität gehen echt nicht in Features, und in einem kapazitätsbeschränkten Team ist dies eine echte Reduktion der Feature-Lieferung, die anerkannt statt weggeredet werden muss.
- Budgets werden unter Druck zuerst gekürzt. Ein Budget, das nur in ruhigen Perioden überlebt, bietet wenig Wert, da sich Verkommen genau während der arbeitsreichen beschleunigt.
- Ohne evidenzbasierte Auswahl kann das Budget für die befriedigendsten statt die wertvollsten Verbesserungen ausgegeben werden — ein ordentliches Modul neu schreiben, während das echt gefährliche unangetastet bleibt.
- Erträge sind verzögert und diffus, was das Budget schwer gegen einen Stakeholder zu verteidigen macht, der Zurechnung für dieses Quartal will.
- Ein Budget kann zum Alibi werden: Die Existenz von zehn Prozent Verbesserungskapazität kann genutzt werden, um zu argumentieren, dass die Probleme des Systems gehandhabt werden, während der tatsächliche Bedarf um ein Mehrfaches größer ist.

## How It Could Be

Ein Team, das ein Fertigungsausführungssystem pflegte, hatte einen Build, der achtunddreißig Minuten dauerte, eine Testsuite, die intermittierend fehlschlug, und vier Abhängigkeiten jenseits ihres Support-Endes. Jeder Versuch, dies anzugehen, verlor zwei Jahre lang gegen den Feature-Backlog. Ihr Engineering Manager verhandelte ein fünfzehnprozentiges Verbesserungsbudget mit dem Produktdirektor auf sechsmonatiger Probebasis, mit der expliziten Bedingung, dass Ergebnisse jeden Monat berichtet würden. Im ersten Quartal senkte das Team den Build auf neun Minuten, quarantänierte und behob die zwölf flakigsten Tests und aktualisierte zwei Abhängigkeiten. Das berichtete Ergebnis, das das Budget dauerhaft sicherte, war nichts davon direkt: Es war, dass die Zahl der pro Monat abgeschlossenen Feature-Posten um achtzehn Prozent stieg, weil ein neunminütiger Build veränderte, wie oft Entwickler integrieren konnten.

Ein anderes Team nutzte sein Budget strategischer. Statt es über viele kleine Aufräumarbeiten zu verteilen, verbrachten sie zwei aufeinanderfolgende Quartale an einem einzigen Subsystem, das ihre Änderungshäufigkeits- und Defektdaten als verantwortlich für etwa vierzig Prozent der Produktionsvorfälle identifizierten, obwohl es nur etwa acht Prozent der Codebasis ausmachte. Die Arbeit war unglamourös — Extraktion einer 3.000-Zeilen-Klasse, Hinzufügen von Characterization Tests und Entfernung dreier Schichten angesammelter Workarounds. Vorfälle aus diesem Subsystem fielen im folgenden Jahr auf nahezu null, und die Bereitschaftsrotation, die bei zwei Kündigungen ein bedeutender Faktor gewesen war, hörte auf, ein Grund zu sein, dass Menschen gingen.
