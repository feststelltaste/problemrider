---
title: Design by Contract
description: Spezifikation von Vorbedingungen, Nachbedingungen und Invarianten für
  überprüfbares Verhalten.
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/design-by-contract/
problems:
- hidden-side-effects
- assumption-based-development
- implementation-rework
- defensive-coding-practices
- circular-references
- difficult-to-understand-code
- suboptimal-solutions
- ripple-effect-of-changes
- complex-implementation-paths
- cognitive-overload
- integer-overflow-underflow
- null-pointer-dereferences
- improper-event-listener-management
- stack-overflow-errors
- entity-attribute-value-overuse
layout: solution
lang: de
en_slug: design-by-contract
related_solutions:
- slug: contract-testing
  similarity: 0.8
- slug: solid-principles
  similarity: 0.75
- slug: clean-code
  similarity: 0.7
- slug: consumer-driven-contracts
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
---

## Description

Design by Contract macht die Vorbedingungen, Nachbedingungen und Invarianten einer Funktion explizit und — idealerweise — ausführbar, und ersetzt die impliziten Annahmen darüber, was Aufrufer liefern müssen und was eine Funktion garantiert, die sich über Jahre undokumentierter Entwicklung still in Legacy-Code anhäufen. Weil diese Annahmen meist nie aufgeschrieben werden, sind sich unterschiedliche Aufrufer in einer Legacy-Codebasis oft uneinig darüber — ein Aufrufer validiert Eingaben vorab, die die Funktion auch validiert, ein anderer überspringt Validierung, die die Funktion still als erledigt annimmt —, und die resultierende Diskrepanz ist genau die Fehlerklasse, die diese Praxis sofort aufdeckt, als Assertion-Fehler, statt als subtiler Produktionsvorfall Monate später. Verträge schrittweise auf den Code einzuführen, den ein Team bereits anfasst, statt zu versuchen, sie überall auf einmal nachzurüsten, verhindert, dass die Praxis zu ihrer eigenen Wartungslast wird.

## How to Apply ◆

> In Legacy-Systemen ersetzt Design by Contract die impliziten Annahmen, die sich über Jahre undokumentierter Entwicklung angehäuft haben, durch explizite, verifizierbare Vereinbarungen darüber, was jede Komponente erwartet, garantiert und aufrechterhält. Dies macht die versteckten Regeln der Legacy-Codebasis sichtbar und durchsetzbar.

- Beginnen Sie damit, Verträge für die am häufigsten missverstandenen Funktionen in der Legacy-Codebasis zu dokumentieren — jene, die Entwickler konsequent falsch aufrufen oder die unerwartete Nebeneffekte erzeugen. Formulieren Sie Vorbedingungen (was wahr sein muss, bevor die Funktion aufgerufen wird), Nachbedingungen (was die Funktion nach der Ausführung garantiert) und Invarianten (was während der gesamten Ausführung der Funktion wahr bleibt).
- Nutzen Sie Assertion-Bibliotheken oder sprachnative Vertragsmechanismen (Java `assert`, Python `assert`, C# Code Contracts oder Bibliotheken wie `icontract` für Python oder `valid4j` für Java), um Verträge ausführbar statt rein dokumentarisch zu machen. Ausführbare Verträge fangen Verletzungen während Testing und Entwicklung ab, bevor sie Produktion erreichen.
- Ersetzen Sie übermäßiges defensives Codieren durch explizite Vorbedingungsprüfungen an Systemgrenzen. Statt dass jede interne Funktion all ihre Eingaben gegen unmögliche Bedingungen validiert, validieren Sie Eingaben einmal am Eintrittspunkt und nutzen Sie Verträge, um zu garantieren, dass interne Funktionen gültige Daten erhalten. Dies eliminiert den defensiven Ballast, der Geschäftslogik verschleiert.
- Definieren Sie Nachbedingungen, die Funktionsverhalten explizit machen, was direkt das Problem versteckter Nebeneffekte adressiert. Wenn eine Funktion Zustand über ihren Rückgabewert hinaus modifiziert, sollte die Nachbedingung dies explizit deklarieren. Kann eine Nachbedingung nicht sauber formuliert werden, weil die Funktion zu viele Dinge tut, ist das ein Signal, die Funktion in fokussierte Komponenten zu trennen.
- Nutzen Sie Klasseninvarianten, um Probleme mit zirkulären Referenzen zu verhindern, indem Sie den gültigen Zustand deklarieren, den Objekte aufrechterhalten müssen. Eine Invariante, die besagt „die Page-Referenzen eines Documents müssen auf dieses Document und kein anderes zurückverweisen", macht die Eigentümerschaftsbeziehung explizit und bei Verletzung erkennbar.
- Wenden Sie Verträge auf Schnittstellen zwischen Legacy-System-Modulen an, um die Vereinbarungen zu definieren, die zuvor implizit waren. Wenn Modul A Modul B aufruft, spezifiziert der Vertrag genau, was A liefern muss und was B liefern wird, und macht die Wellenwirkung von Änderungen sichtbar: Ändert sich ein Vertrag, müssen alle Aufrufer aktualisiert werden.
- Führen Sie Verträge schrittweise ein, indem Sie sich auf Code konzentrieren, der aktiv modifiziert wird, statt zu versuchen, Verträge über die gesamte Legacy-Codebasis nachzurüsten. Jedes Mal, wenn ein Entwickler eine Funktion anfasst, fügt er den Vertrag hinzu, der das tatsächliche Verhalten der Funktion dokumentiert.
- Nutzen Sie während des Testens entdeckte Vertragsverletzungen als Diagnosewerkzeug: Sie enthüllen Annahmen, die im Legacy-Code eingebettet, aber nie dokumentiert waren, und verweisen oft auf die Ursache langjähriger Fehler.

## Tradeoffs ⇄

> Design by Contract macht die impliziten Regeln eines Legacy-Systems explizit und verifizierbar, erfordert aber Investition in die Definition und Pflege von Verträgen neben dem Code, den sie schützen.

**Vorteile:**

- Eliminiert versteckte Nebeneffekte, indem Entwickler gezwungen werden, alles, was eine Funktion tut, als Teil ihrer Nachbedingung zu deklarieren, was undokumentiertes Verhalten sichtbar und überprüfbar macht.
- Reduziert Implementierungs-Nacharbeit, indem falsche Annahmen früh abgefangen werden: Wenn das Verständnis eines Entwicklers darüber, wie eine Funktion aufgerufen werden sollte, falsch ist, taucht die Vorbedingungsverletzung sofort während der Entwicklung auf, statt nach dem Deployment.
- Ersetzt verschwenderisches defensives Codieren durch gezielte Grenzvalidierung, was Codeumfang und kognitive Last reduziert, während Korrektheitsgarantien erhalten oder verbessert werden.
- Macht die im Legacy-Code eingebetteten Annahmen explizit und dokumentiert, sodass Entwickler, die später zum Team stoßen, das Verhalten von Komponenten aus Verträgen statt durch Lesen und Reverse-Engineering von Implementierungsdetails verstehen können.
- Bietet präzise Dokumentation der Änderungsauswirkung: Wenn sich ein Vertrag ändert, ist die Menge der betroffenen Aufrufer sofort identifizierbar, was die unvorhersehbare Wellenwirkung in einen begrenzten, handhabbaren Umfang verwandelt.

**Kosten und Risiken:**

- Verträge fügen Wartungsoverhead hinzu: Wenn sich die Implementierung ändert, müssen Verträge synchron aktualisiert werden, und veraltete Verträge sind schlimmer als keine Verträge, weil sie falsche Sicherheit bieten.
- Laufzeit-Assertion-Prüfung hat Performance-Kosten, die in performance-kritischen Legacy-Codepfaden inakzeptabel sein können; Verträge in solchen Bereichen müssen möglicherweise in Produktion deaktiviert werden und nur während des Testens aktiv sein.
- Legacy-Code mit tief verworrenen Verhalten kann Verträge haben, die extrem komplex korrekt zu spezifizieren sind, und falsche Verträge erzeugen ein falsches Sicherheitsgefühl, während echte Fehler durchschlüpfen.
- Mit Design by Contract nicht vertraute Teams könnten trivial offensichtliche Verträge schreiben (Vorbedingung: Parameter ist nicht null), die Ballast ohne Wert hinzufügen, statt bedeutungsvoller Verhaltensverträge, die echten Schutz bieten.
- Verträge in Legacy-Code nachzurüsten erfordert das Verständnis des tatsächlichen Verhaltens des Codes, was genau das Problem ist, das Verträge lösen sollen — Verträge in schlecht verstandenem Code zu bootstrappen erfordert zunächst sorgfältiges Charakterisierungstesting.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Design by Contract angewandt wurde, um Klarheit und Korrektheitsgarantien in Legacy-Systeme zu bringen, wo implizite Annahmen wiederkehrende Probleme verursachten.

Ein Zahlungsabwicklungsunternehmen hatte eine `TransactionProcessor.process()`-Methode, die Entwickler in mehreren unterschiedlichen Kontexten aufriefen, jeder mit unterschiedlichen Annahmen über den Zustand des Transaktionsobjekts. Manche Aufrufer erwarteten, dass die Methode die Transaktion zuerst validiert; andere validierten vorab und nahmen an, die Methode würde Validierung überspringen. Keine der Annahmen war dokumentiert, und das Verhalten der Methode hing von einem internen Flag ab, das nicht Teil ihrer öffentlichen Schnittstelle war. Das Team fügte explizite Vorbedingungen hinzu: `transaction.status == VALIDATED` und `transaction.amount > 0`. Nachbedingungen spezifizierten: `transaction.status == PROCESSED` und `auditLog.contains(transaction.id)`. Aufrufer, die Vorbedingungen verletzten, wurden während Integrationstests sofort durch Assertion-Fehler identifiziert, was drei Codepfade enthüllte, die monatelang still ungültige Transaktionen verarbeitet hatten.

Das Legacy-Schadensbearbeitungssystem eines Versicherungsunternehmens hatte anhaltende Probleme mit zirkulären Referenzen zwischen `Claim`- und `Policy`-Objekten. Beide hielten veränderliche Referenzen aufeinander, und je nach Reihenfolge der Operationen konnte ein Claim am Ende auf eine Policy verweisen, die auf einen anderen Claim verwies. Das Team führte Klasseninvarianten auf beiden Objekten ein: Eine `Claim`-Invariante besagte, dass `this.policy.claims.contains(this)` immer gelten muss, und eine `Policy`-Invariante besagte, dass für jeden Claim in ihrer Sammlung `claim.policy == this` gilt. Diese Invarianten markierten sofort die Initialisierungssequenz, die verwaiste Referenzen erzeugte, ein Fehler, der zwei Jahre lang intermittierende Berichtsinkonsistenzen verursacht hatte.

Ein Logistikunternehmen litt unter chronischer Implementierungs-Nacharbeit, weil Entwickler Annahmen über Sendungszustandsübergänge trafen, die nicht zu tatsächlichen Geschäftsregeln passten. Ein Entwickler würde „Sendung als zugestellt markieren" implementieren unter der Annahme, dass die Sendung im Zustand „in Transit" ist, aber manche Sendungen konnten direkt aus dem Lager zugestellt werden, ohne je durch „in Transit" zu übergehen. Das Team definierte einen Zustandsmaschinenvertrag für die `Shipment`-Klasse mit expliziten Vorbedingungen für jeden Zustandsübergang. Der Vertrag dokumentierte, dass `markDelivered()` `status in [IN_TRANSIT, AT_WAREHOUSE]` erforderte, nicht nur `IN_TRANSIT`. Mit diesen Verträgen konnten Entwickler genau sehen, welche Übergänge gültig waren, bevor sie eine einzige Zeile Implementierungscode schrieben, was Nacharbeit durch falsche Zustandsannahmen um über 60 Prozent reduzierte.
