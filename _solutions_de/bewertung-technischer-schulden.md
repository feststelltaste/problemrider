---
title: Bewertung technischer Schulden
description: Untersuchung eines Bereichs im Detail, zeitlich begrenzt, mit
  einem schriftlichen Bild dessen, was dort tatsächlich falsch ist —
  ersetzt allgemeine Sorge durch konkrete Befunde.
category:
- Code
- Architecture
- Process
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- brittle-codebase
- increasing-brittleness
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- maintenance-paralysis
- modernization-strategy-paralysis
- fear-of-change
- legacy-system-documentation-archaeology
- maintenance-bottlenecks
- difficult-to-understand-code
- analysis-paralysis
- accumulation-of-workarounds
- large-estimates-for-small-changes
- maintenance-cost-increase
- refactoring-avoidance
- workaround-culture
- core-modification-of-standard-software
- low-code-customization-sprawl
layout: solution
lang: de
en_slug: technical-debt-assessment
related_solutions:
- slug: technical-debt-backlog
  similarity: 0.8
- slug: debt-accrual-analysis
  similarity: 0.8
- slug: debt-classification
  similarity: 0.75
- slug: code-hotspot-analysis
  similarity: 0.75
- slug: debt-remediation-estimation
  similarity: 0.75
- slug: code-metrics
  similarity: 0.7
---

## Description

Eine Bewertung technischer Schulden ist eine zeitlich begrenzte, strukturierte Untersuchung eines abgegrenzten Bereichs eines Systems, die in einem schriftlichen Dokument endet, das angibt, was dort falsch ist, wie schlimm jede Sache ist und was es kosten würde, sie zu adressieren. Sie unterscheidet sich von einem Metrik-Dashboard, das Zahlen ohne Urteil berichtet, und von einem Architektur-Review, das periodisch und breit ist. Dies ist bewusst eng und tief: ein Subsystem, ein bis drei Wochen, ein Dokument. Ihr Zweck ist es, einen allgemeinen Zustand — "dieses Modul ist ein Albtraum" — in eine endliche Liste benannter Befunde zu verwandeln. Diese Umwandlung ist der Punkt. Teams und Manager erleben Legacy-Schulden als eine unbegrenzte Sorge, und ein unbegrenztes Problem kann nicht geplant, finanziert oder priorisiert werden. Eine Liste von elf spezifischen Befunden, jeder mit einer Größe, ist ein Problem, an dem gearbeitet werden kann, selbst wenn die Liste alarmierend ist.

## How to Apply ◆

> Der Bereich, über den alle vermeiden, konkret zu sprechen, ist üblicherweise der, bei dem eine Bewertung am meisten zurückgibt, genau weil niemand hingesehen hat.

- **Wählen Sie einen abgegrenzten Bereich**, nicht das System. Ein Subsystem, ein Modul, ein Datenfluss. Die Bewertung von allem produziert ein Dokument, das niemand liest, und Befunde, die zu allgemein sind, um darauf zu handeln. Hotspot-Daten — Änderungshäufigkeit gekreuzt mit Fehlerbeteiligung — wählen den Bereich, wenn die Intuition unsicher ist.
- **Setzen Sie eine strikte Zeitbox**, ein bis drei Wochen je nach Größe, und geben Sie im Voraus an, dass die Ausgabe ein Bild ist statt eines vollständigen Inventars. Bewertungen ohne festes Ende wachsen, bis sie aufgegeben werden.
- **Nutzen Sie mehrere Linsen und protokollieren Sie, welche was fand**: Lesen des Codes, der Änderungshistorie, des Vorfallprotokolls, der Testabdeckung, der Abhängigkeitsstruktur und Interviews mit denen, die dort arbeiten. Jede bringt Dinge zutage, die die anderen übersehen, und besonders die Änderungshistorie offenbart Probleme, die keine Codelektüre findet.
- **Geben Sie jeden Befund konkret an**: was er ist, wo er ist, was er heute kostet, was passiert, wenn nichts getan wird, und eine grobe Größe zur Adressierung. Ein Befund ohne Kosten und Größe ist eine Beobachtung, und Beobachtungen werden nicht finanziert.
- **Trennen Sie, was schmerzt, von dem, was nur hässlich ist.** Das meiste, was eine Bewertung berichten könnte, ist ästhetisch und kostet nichts. Es zu berichten verwässert die Befunde, die zählen, und ist der Grund, warum Bewertungen den Ruf erwerben, Wunschlisten zu produzieren.
- **Beziehen Sie ein, was funktioniert.** Eine Bewertung, die nur Probleme findet, liest sich wie eine Anklage der Personen, die es gebaut haben, was die nächste schwerer zu arrangieren macht und das Team defensiv statt kollaborativ macht.
- **Lassen Sie es jemanden von außerhalb des Bereichs mit jemandem von innerhalb tun.** Der Außenstehende stellt die Fragen, die Vertrautheit unterdrückt hat; der Innenstehende verhindert, dass der Außenstehende bewusste Entscheidungen als Fehler missversteht.
- **Schreiben Sie es für zwei Publikum.** Eine einseitige Zusammenfassung in Kosten- und Risikobegriffen, und die detaillierten Befunde darunter. Eine Bewertung, die nur von Ingenieuren gelesen werden kann, kann die Aufgabe nicht erfüllen, die Schulden für diejenigen greifbar zu machen, die die Arbeit finanzieren.
- **Enden Sie mit einer empfohlenen Reihenfolge**, nicht nur einer Liste. Welche drei Befunde zuerst, und warum. Eine Liste von elf Befunden ohne Reihenfolge gibt das Priorisierungsproblem an denjenigen zurück, der die Bewertung in Auftrag gegeben hat.
- **Bewerten Sie nach der Sanierung erneut**, um zu prüfen, ob die Befunde tatsächlich geschlossen wurden. Bewertungen, die nie wieder aufgegriffen werden, werden zu historischen Dokumenten, die ein System beschreiben, das nicht mehr existiert.

## Tradeoffs ⇄

> Eine tiefe Bewertung verwandelt vage Sorge in eine endliche, planbare Liste, auf Kosten echten Aufwands und dem Risiko, ein Dokument zu produzieren, das einmal gelesen und dann weggelegt wird.

**Vorteile:**

- Das Problem wird begrenzt. Eine benannte Liste von Befunden kann priorisiert, dimensioniert und finanziert werden; ein allgemeiner Zustand kann das nicht, und das ist üblicherweise der eigentliche Blocker.
- Angst wird proportional. Bewertungen finden routinemäßig, dass ein gefürchtetes Subsystem drei echte Probleme hat und eine große Menge nur unangenehmen Codes, was ändert, wie das Team ihm begegnet.
- Befunde tragen Kosten und Größen, was es erlaubt, dass Schuldenarbeit in eine Priorisierungsdiskussion auf denselben Bedingungen eintritt wie alles andere.
- Die schriftliche Aufzeichnung überlebt Personalwechsel, sodass das Verständnis nicht verloren geht, wenn die Person, die untersucht hat, weiterzieht.
- Der Multi-Linsen-Ansatz bringt Probleme zutage, die kein einzelnes Werkzeug erkennt, besonders die, die nur in der Änderungshistorie oder dem Vorfallprotokoll sichtbar sind.

**Kosten und Risiken:**

- Ein bis drei Wochen fähiger Personen produzieren keine funktionierende Software, und diese Kosten werden sofort gespürt.
- Bewertungen enden häufig als Dokumente, auf die niemand reagiert, was den Aufwand verschwendet und der Organisation beibringt, dass Bewertung Theater ist.
- Befunde können sich wie Kritik an den Personen lesen, die den Code geschrieben haben, und in einer Schuldkultur wird die Bewertung Widerstand erfahren oder still entschärft werden.
- Ein abgegrenzter Umfang bedeutet, dass alles außerhalb davon unbewertet bleibt, und das schlimmste Problem könnte im Bereich liegen, der nicht gewählt wurde.
- Das Bild wird veraltet. In einem aktiv entwickelten Bereich ist eine Bewertung eine Momentaufnahme mit einer Haltbarkeit von Monaten.

## How It Could Be

Ein Team beschrieb sein Abrechnungssubsystem als "den Teil, wo Dinge schiefgehen" und war zwei Jahre lang nicht in der Lage gewesen, irgendeine Verbesserung finanziert zu bekommen, weil jeder Vorschlag darauf hinauslief, um Zeit zu bitten, eine schlechte Sache besser zu machen. Sie bewerteten es über zwei Wochen, zwei Personen, vier Linsen. Das Ergebnis waren elf Befunde, von denen vier als heute kostend bewertet wurden: eine duplizierte Steuerberechnung, die zwischen ihren zwei Kopien auseinandergedriftet war, eine Retry-Schleife, die eine spezifische Fehlerklasse still verschluckte, ein Fehlen von Tests um die Proration-Logik, und ein geplanter Job, dessen Fehlschlag nicht alarmiert wurde. Die Größen reichten von zwei Tagen bis sechs Wochen. Sieben weitere Befunde wurden als hässlich, aber harmlos protokolliert. Die vier kostspieligen Befunde wurden innerhalb eines Monats finanziert — nicht weil das Subsystem weniger schlecht geworden war, sondern weil die Anfrage nun für ein spezifisches sechswöchiges Arbeitspaket war, statt für eine offene Verbesserung.

Der Proportionalitätseffekt überraschte das Team mehr als die Finanzierung. Ihr kollektives Gefühl war gewesen, dass das Subsystem gleichmäßig gefährlich sei und dass das Anfassen jedes Teils davon riskant sei. Die Bewertung fand, dass ungefähr 70 Prozent davon mühsam, aber unkompliziert war, und dass sich das Risiko in zwei Dateien konzentrierte. Entwickler, die um das gesamte Subsystem herum navigiert hatten, begannen, normal in der sicheren Mehrheit davon zu arbeiten. Die einseitige Zusammenfassung — vier Befunde, ihre monatlichen Kosten und eine empfohlene Reihenfolge — war auch das erste Dokument über dieses Subsystem, das der Finanzdirektor je gelesen hatte, und es war das, was das Gespräch von "die Entwicklung will Dinge neu schreiben" zu einer Diskussion über Sequenzierung wandelte.
