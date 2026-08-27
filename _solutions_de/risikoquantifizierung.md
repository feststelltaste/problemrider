---
title: Risikoquantifizierung
description: Ausdruck des Legacy-Risikos als erwarteten Verlust in Geld —
  Wahrscheinlichkeit mal Auswirkung — sodass vermiedener Schaden in einer
  Finanzierungsentscheidung mit Umsatz konkurrieren kann.
category:
- Business
- Management
- Security
problems:
- modernization-roi-justification-failure
- difficulty-quantifying-benefits
- obsolete-technologies
- technology-lock-in
- legacy-skill-shortage
- single-points-of-failure
- knowledge-silos
- regulatory-compliance-drift
- increasing-brittleness
- system-stagnation
- system-outages
- vendor-dependency
- high-defect-rate-in-production
- competitive-disadvantage
- deployment-risk
- high-maintenance-costs
- invisible-nature-of-technical-debt
- missing-rollback-strategy
- project-resource-constraints
- technology-stack-fragmentation
- vendor-dependency-entrapment
- high-technical-debt
- implementation-partner-dependency
- retention-obligations-block-change
- upgrade-blocked-by-customization
- voided-vendor-support
layout: solution
lang: de
en_slug: risk-quantification
related_solutions:
- slug: risk-analysis
  similarity: 0.75
- slug: total-cost-of-ownership-transparency
  similarity: 0.7
- slug: cost-of-delay
  similarity: 0.7
- slug: requirements-analysis
  similarity: 0.7
- slug: functional-spike
  similarity: 0.7
- slug: knowledge-sharing-practices
  similarity: 0.65
---

## Description

Risikoquantifizierung drückt ein Risiko als erwarteten jährlichen Verlust aus — wie wahrscheinlich es eintritt, multipliziert mit dem, was es kosten würde, wenn es einträte —, sodass seine Vermeidung gegen andere Verwendungen von Geld verglichen werden kann. Sie adressiert den spezifischen Fehler, der die meisten Legacy-Modernisierungsfälle versenkt: Der Wert der Arbeit ist größtenteils vermiedener Schaden, und vermiedener Schaden hat keine natürliche Einheit, sodass er qualitativ beschrieben wird, während der Alternativvorschlag mit einer Umsatzzahl ankommt. Qualitativ verliert in einer Finanzierungsentscheidung immer gegen Quantitativ, unabhängig davon, was wichtiger ist. Die Technik beansprucht keine Präzision. Eine aus festgelegten Annahmen gebaute Spanne ist keine genaue Vorhersage dessen, was passieren wird, und sie versucht das auch nicht: Sie ist eine Übersetzung, die „dies ist gefährlich" in eine Form konvertiert, die der Entscheidungsprozess tatsächlich abwägen kann.

## How to Apply ◆

> Legacy-Risiko ist ungewöhnlich empfänglich dafür, weil die Fehlermodi bekannt sind — die Organisation hat oft bereits kleinere Versionen davon erlebt.

- **Benennen Sie das spezifische Fehlerszenario**, nicht den allgemeinen Zustand. „Der Mainframe ist alt" kann nicht quantifiziert werden. „Der eine verbleibende COBOL-Entwickler verlässt das Unternehmen, und ein Produktionsdefekt im Abrechnungs-Batch braucht sechs Wochen statt zwei Tage zur Behebung" kann es, weil beide Begriffe schätzbar sind.
- **Schätzen Sie Wahrscheinlichkeit und Auswirkung als Spannen**, mit aufgeschriebener Begründung. Präzision ist nicht das Ziel, und so zu tun als hätte man sie, lädt zum Angriff ein; „irgendwo zwischen 10 und 25 Prozent in einem gegebenen Jahr" mit einer festgelegten Grundlage ist verteidigbarer als eine einzelne selbstbewusste Zahl.
- **Bauen Sie die Auswirkung aus Komponenten**, die die Organisation bereits bepreist: Ausfallstunden mal Umsatz pro Stunde, Wiederherstellungsaufwand, vertragliche Vertragsstrafen, regulatorische Bußgelder, Benachrichtigungs- und Behebungskosten, und wo relevant die Kosten abwandernder Kunden. Jede Komponente kann unabhängig geprüft werden.
- **Nutzen Sie die eigene Geschichte der Organisation.** Vergangene Vorfälle sind die beste verfügbare Evidenz für beide Begriffe, und die meisten Legacy-Risiken haben bereits Beinahe-Unfälle oder kleinere Instanzen produziert, die behandelt und vergessen wurden. Diese zu rekonstruieren ist üblicherweise der überzeugendste Teil der Analyse.
- **Modellieren Sie das wachsende Risiko.** Anders als die meisten Risiken steigt Legacy-Risiko generell über die Zeit — ein erodierender Kompetenzpool, ein näherkommendes Support-Ende-Datum, sich anhäufendes Datenvolumen. Den erwarteten Verlust als Kurve statt als Zahl zu zeigen ist es, was das Timing-Argument macht.
- **Präsentieren Sie das Restrisiko nach der vorgeschlagenen Arbeit**, nicht nur das aktuelle Risiko. Der Nutzen ist die Differenz zwischen den beiden, und ein Vorschlag, der impliziert, das Risiko gehe auf null, wird von niemandem Erfahrenem geglaubt.
- **Beziehen Sie die Funktionen ein, die dies bereits tun.** Finanzen, Versicherung und Risikomanagement haben etablierte Methoden und, wichtiger, etablierte Glaubwürdigkeit. Eine gemeinsam mit ihnen produzierte Zahl wird sehr anders behandelt als eine, die vom Engineering allein produziert wurde.
- **Halten Sie die Annahmen sichtbar und getrennt von der Schlussfolgerung**, sodass ein Skeptiker mit einer Eingabe argumentieren kann, statt die Ausgabe abzulehnen. Das produktivste Ergebnis einer solchen Analyse ist häufig eine Meinungsverschiedenheit über eine spezifische Wahrscheinlichkeit, was ein Gespräch ist, das konvergieren kann.
- **Quantifizieren Sie nicht alles.** Manche Risiken sind echt nicht quantifizierbar, und eine Zahl darauf zu erzwingen produziert Werte, die diejenigen diskreditieren, die ordnungsgemäß gemacht wurden.

## Tradeoffs ⇄

> Quantifizierung lässt Risikoreduktion auf gleichen Bedingungen um Finanzierung konkurrieren, zum Preis falscher Präzision, anfechtbarer Annahmen und einer echten Chance, öffentlich falsch zu liegen.

**Vorteile:**

- Vermiedener Schaden erwirbt eine Einheit und kann gegen umsatzgenerierende Alternativen verglichen werden, was der einzige Weg ist, wie Risikoreduktionsarbeit eine Priorisierungsentscheidung gewinnt.
- Die steigende Risikokurve macht das Timing-Argument, das eine statische Beschreibung nicht kann, und verwandelt „wir sollten das tun" in „der erwartete Verlust übersteigt ab nächstem Jahr die Kosten der Behebung".
- Meinungsverschiedenheiten werden spezifisch und lösbar — über eine Wahrscheinlichkeit oder eine Auswirkungskomponente — statt ein allgemeiner Zusammenstoß von Intuitionen zu sein.
- Die Rekonstruktion vergangener Beinahe-Unfälle deckt häufig auf, dass die Organisation bereits erhebliche Beträge für ein Risiko gezahlt hat, das sie für hypothetisch hielt.
- Die Analyse ist wiederverwendbar. Sobald das Modell existiert, zeigt jährliches erneutes Ausführen, ob sich das Risikoprofil verbessert, was selbst ein Managementinstrument ist.

**Kosten und Risiken:**

- Wahrscheinlichkeitsschätzungen für seltene Ereignisse sind echt unzuverlässig, und die Arithmetik gibt ihnen einen Anschein von Rigorosität, den sie nicht haben.
- Eine einzelne quantifizierte Zahl lädt die Antwort ein, dass das Risiko erschwinglich sei, was eine legitime Entscheidung ist, aber möglicherweise nicht die, die die Analyse unterstützen sollte.
- Die Übung erfordert echten Aufwand und spezialisierten Input, und sie kann mehr Zeit verbrauchen, als die Entscheidung für kleinere Risiken rechtfertigt.
- Quantifizierte Risiken konkurrieren miteinander, und eines, das echt schwerwiegend, aber schwer zu schätzen ist, wird gegen eines verlieren, das moderat und leicht zu schätzen ist.
- Wenn das quantifizierte Risiko nie eintritt, kann die Analyse rückwirkend als Panikmache charakterisiert werden, was die nächste erschwert.

## How It Could Be

Der Zahlungsabgleich einer Organisation lief auf einer Plattform mit einem verbleibenden Entwickler, der sie verstand, im Alter von 61 Jahren. Drei Modernisierungsvorschläge waren als unzureichend gerechtfertigt abgelehnt worden. Der vierte quantifizierte das spezifische Szenario statt die allgemeine Sorge zu beschreiben. Die Wahrscheinlichkeit, dieses Wissen innerhalb von drei Jahren zu verlieren — Ruhestand, Krankheit oder Kündigung — wurde auf 60 bis 80 Prozent geschätzt, unter Nutzung der eigenen versicherungsmathematischen Annahmen der Organisation für eine Person dieses Alters und dieser Betriebszugehörigkeit. Die Auswirkung wurde aus vier Komponenten gebaut: geschätzte 4 bis 9 Monate, um das Wissen extern wiederaufzubauen, Auftragnehmersätze für diesen Zeitraum, der Abgleichsrückstand, der sich währenddessen mit einer Rate anhäufen würde, die sie aus einer zweiwöchigen Abwesenheit im Vorjahr messen konnten, und eine regulatorische Berichtspflicht, die mit einer festgelegten Strafe verfehlt würde. Der erwartete Verlust belief sich auf 1,4 bis 3,1 Millionen Euro gegen Modernisierungskosten von 900.000 Euro. Die Arbeit wurde innerhalb eines Monats finanziert.

Die Restrisiko-Rahmung war das, was den Fall die Prüfung überstehen ließ. Der Vorschlag behauptete nicht, dass das Risiko eliminiert würde — der Ersatz würde Wissen immer noch konzentrieren, nur weniger schwerwiegend, in einer Technologie mit einem größeren verfügbaren Kompetenzpool. Er nannte einen Restrisiko-erwarteten Verlust von 300.000 bis 600.000 Euro und benannte die weiteren Maßnahmen, die ihn reduzieren würden. Ein CFO, der die vorherigen drei Vorschläge abgelehnt hatte, kommentierte, dies sei der erste, der nicht wie Werbung geklungen habe. Die zweiwöchige Abwesenheit im Vorjahr, rekonstruiert und als Teil der Analyse bepreist, stellte sich heraus, die Organisation ungefähr 70.000 Euro an Auftragnehmerzeit und verzögerter Berichterstattung gekostet zu haben — ein Vorfall, der behandelt, in Rechnung gestellt und nie von irgendjemandem mit dem zugrunde liegenden Risiko verbunden worden war.
