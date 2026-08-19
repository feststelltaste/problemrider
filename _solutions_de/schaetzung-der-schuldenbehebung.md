---
title: Schätzung der Schuldenbehebung
description: Aufwandsschätzung für jeden Schuldenposten, sodass die Summe zu einer
  endlichen Zahl wird — denn ein ungeschätztes Problem lässt sich nicht planen und
  fühlt sich unendlich an.
category:
- Code
- Management
- Process
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- maintenance-paralysis
- modernization-strategy-paralysis
- large-estimates-for-small-changes
- planning-credibility-issues
- refactoring-avoidance
- analysis-paralysis
- budget-overruns
- accumulation-of-workarounds
- fear-of-change
- brittle-codebase
- poor-test-coverage
- core-modification-of-standard-software
layout: solution
lang: de
en_slug: debt-remediation-estimation
related_solutions:
- slug: technical-debt-backlog
  similarity: 0.8
- slug: debt-classification
  similarity: 0.8
- slug: technical-debt-assessment
  similarity: 0.75
- slug: debt-accrual-analysis
  similarity: 0.75
- slug: functional-debt-management
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
---

## Description

Schätzung der Schuldenbehebung setzt eine grobe Aufwandszahl auf jeden bekannten Schuldenposten und daraus eine Summe. Die Zahl ist weniger wichtig als die Tatsache, überhaupt eine zu haben. Ein ungeschätztes Problem wird als unendlich erlebt, und ein unendliches Problem erzeugt zwei Reaktionen, beide schlecht: Lähmung, weil es keinen Sinn hat, etwas zu beginnen, das nicht abgeschlossen werden kann, und Verleugnung, weil es leichter ist, ein Problem, das nicht gelöst werden kann, nicht anzuschauen. Beides ist in Legacy-Teams verbreitet, und beides löst sich auf, wenn sich die Summe als Zahl herausstellt. Die Schätzungen müssen nicht genau sein — sie müssen ehrlich über ihre eigene Unsicherheit sein und konsistent genug, dass Posten vergleichbar sind. Häufig ist die Summe kleiner als die Furcht nahelegte, und wo sie echt groß ist, weiß die Organisation zumindest jetzt, was sie zu unterlassen wählt.

## How to Apply ◆

> Die nützlichste Zahl, die eine Bewertung liefert, ist nicht die Summe, sondern die Entdeckung, dass die drei schlimmsten Posten den Großteil davon ausmachen — und dass sie einzeln endlich sind.

- **Schätzen Sie auf einer groben Skala**, nicht in Stunden: ein paar Tage, ein paar Wochen, ein Monat oder zwei, ein Quartal oder mehr. Präzision hier ist trügerisch, und grobe Kategorien sind schneller zu erstellen, leichter zu vereinbaren und ehrlicher über das, was bekannt ist.
- **Schätzen Sie nur die Schulden, die eine Behebung wert sind.** Alle 187 Posten in einem Backlog zu bemessen ist Verschwendung; die zinstragende Teilmenge zu bemessen braucht einen Bruchteil der Zeit und ist das, was die Entscheidung informiert.
- **Schätzen Sie den kleinsten sicheren Schritt**, nicht den idealen Endzustand. „Was würde es brauchen, damit das aufhört wehzutun" ist meist ein Bruchteil von „was würde es brauchen, das richtig zu machen", und beides zu vermengen ist der Grund, warum Schuldenposten Schätzungen bekommen, die garantieren, dass sie nie genehmigt werden.
- **Beziehen Sie das Sicherheitsnetz in die Schätzung ein.** Die Behebung von Legacy-Code erfordert meist zuerst Charakterisierungstests, und Schätzungen, die dies auslassen, sind um einen großen Faktor falsch. Dies ist der häufigste einzelne Grund, warum Schuldenschätzungen überschritten werden.
- **Geben Sie eine Spanne mit der Begründung an**, und seien Sie explizit, wo Sie es nicht wissen. Ein mit „zwei Wochen bis zwei Monate, weil wir nicht wissen, wie viele Konsumenten von dieser Schnittstelle abhängen" geschätzter Posten lädt zu der günstigen Untersuchung ein, die dies einengen würde — was oft der richtige nächste Schritt ist.
- **Verknüpfen Sie die Schätzung mit den Kosten, die sie entfernt.** Ein Posten, der vier Tage im Monat kostet, damit zu leben, und zwei Wochen, ihn zu beheben, amortisiert sich in unter vier Monaten; dieselbe Korrektur an einem ruhenden Posten amortisiert sich nie. Das Zahlenpaar macht das Argument, nicht eine der beiden allein.
- **Veröffentlichen Sie die Summe und ihre Verteilung.** Eine Summe von „etwa acht bis vierzehn Entwicklermonate, wovon zwei Posten die Hälfte ausmachen" ist eine Managementaussage. Sie offenbart auch häufig, dass die Situation weniger katastrophal ist, als alle annahmen.
- **Schätzen Sie nach jeder Behebung neu**, mittels dem, was sie tatsächlich gekostet hat. Legacy-Behebungsschätzungen verbessern sich dramatisch nach ein paar echten Datenpunkten, und die frühen sind meist um einen konsistenten Faktor zu optimistisch, der es wert ist, gemessen zu werden.
- **Lassen Sie die Schätzung nicht zu einer Zusage werden.** Dies sind Bemessungszahlen für Priorisierung, und sie als Lieferversprechen zu behandeln wird das Team dazu bringen, sie aufzublähen, bis sie aufhören, nützlich zu sein.

## Tradeoffs ⇄

> Bemessung verwandelt eine unbegrenzte Furcht in einen endlichen Plan, auf Kosten von Schätzaufwand und Schätzungen, die auf Arten falsch sein werden, die dem Team vorgehalten werden können.

**Vorteile:**

- Das Problem wird endlich, was die Voraussetzung ist, es zu planen, zu finanzieren oder bewusst zu entscheiden, es nicht zu tun.
- Amortisation wird berechenbar, wenn die Schätzung mit den laufenden Kosten gepaart wird, was Schuldenarbeit erlaubt, auf Basis von Belegen zu konkurrieren.
- Die Verteilung zeigt meist, dass eine kleine Anzahl von Posten die Summe dominiert, was eine überwältigende Liste in eine kurze verwandelt.
- Sowohl Lähmung als auch Verleugnung schwächen sich ab, weil es jetzt etwas zu beginnen gibt statt eines endlosen Zustands, den man erdulden muss.
- Die Schätzgenauigkeit verbessert sich über die Zeit, während echte Behebungsdaten sich anhäufen, was jeden nachfolgenden Plan verbessert.

**Kosten und Risiken:**

- Legacy-Behebungsschätzungen sind echt unzuverlässig, weil die Arbeit regelmäßig Abhängigkeiten aufdeckt, von denen niemand wusste — was die Natur der geschätzten Sache ist.
- Schätzungen werden als Zusagen behandelt, unabhängig davon, wie sie gekennzeichnet sind, und das Team zahlt für die Überschreitung.
- Bemessung nimmt Zeit von Menschen, die bereits die Engpassressource sind, und sie produziert keine funktionierende Software.
- Eine große ehrliche Summe kann den Glauben einer Organisation bestätigen, dass die Situation hoffnungslos ist, was den entgegengesetzten Effekt des beabsichtigten erzeugt.
- Den kleinsten sicheren Schritt zu schätzen kann untertreiben, was letztlich benötigt wird, was eine Sequenz partieller Behebungen hinterlässt, die nie einen guten Zustand erreicht.

## How It Could Be

Die technischen Schulden eines Teams wurden in jeder Planungsdiskussion als überwältigend beschrieben, und intern hieß es, das System werde „irgendwann eine Neuschreibung" brauchen. Sie klassifizierten ihren Backlog, fanden 24 zinstragende Posten und bemaßen diese über zwei Tage in vier groben Kategorien. Die Summe betrug etwa neun bis sechzehn Entwicklermonate. Zwei Posten — die duplizierte Preislogik und das Fehlen jeglicher Tests rund um den Abrechnungs-Batch — machten etwa die Hälfte davon aus. Die Reaktion im Raum war hörbare Erleichterung: neun bis sechzehn Entwicklermonate war eine große Zahl, aber eine begreifbare, und erheblich kleiner als die Neuschreibung, die alle halb angenommen hatten. Die beiden dominanten Posten wurden über die folgenden zwei Quartale als ein einziges Stück Arbeit finanziert.

Größen mit laufenden Kosten zu paaren ordnete die Liste auf eine Weise um, die niemand erwartete. Ein Posten, den alle beheben wollten — ein schlecht strukturiertes Berichtsmodul, geschätzt auf sechs Wochen — stellte sich heraus, etwa einen halben Tag im Monat zu kosten, damit zu leben, eine Amortisation von etwa zwanzig Jahren. Ein Posten, für den niemand plädiert hatte, ein fehlender Index und eine schlecht geformte Abfrage, wurde auf drei Tage geschätzt und kostete etwa drei Tage im Monat in Support-Bearbeitung von Timeouts. Er wurde in dieser Woche erledigt. Die spätere Zusammenfassung des Teams zu der Übung war, dass sie vier Jahre damit verbracht hatten, Schulden danach zu priorisieren, wie sehr sie sie störten, und dass die Störung fast keine Beziehung zu den Kosten hatte.
