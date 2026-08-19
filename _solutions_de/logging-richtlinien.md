---
title: Logging-Richtlinien
description: Vereinbarung, was auf welcher Ebene geloggt wird, was niemals
  geloggt werden darf und wie lange es aufbewahrt wird — sodass Logs zu einem
  Diagnosewerkzeug werden statt zu einem Volumenproblem.
category:
- Code
- Operations
problems:
- excessive-logging
- log-spam
- logging-configuration-issues
- log-injection-vulnerabilities
- monitoring-gaps
- inadequate-error-handling
- debugging-difficulties
- operational-overhead
- excessive-disk-io
- resource-waste
layout: solution
lang: de
en_slug: logging-guidelines
related_solutions:
- slug: logging-and-monitoring
  similarity: 0.75
- slug: observability-and-monitoring
  similarity: 0.7
- slug: logging
  similarity: 0.7
- slug: error-logging
  similarity: 0.65
- slug: audit-trail-management
  similarity: 0.65
- slug: data-flow-control
  similarity: 0.6
---

## Description

Logging-Richtlinien sind eine kurze Vereinbarung darüber, was ein System in seine Logs schreibt: welche Ereignisse welchen Schweregrad rechtfertigen, welchen Kontext jeder Eintrag tragen muss, was nie erscheinen darf und wie lange Einträge aufbewahrt werden. Sie adressieren einen Fehlermodus, der in langlebigen Systemen fast universell ist und selten als eigenständiges Problem behandelt wird. Logging wird von vielen Menschen über viele Jahre inkrementell hinzugefügt, ohne gemeinsame Konvention, und konvergiert auf das Schlimmste beider Ergebnisse: enormes Volumen und keinen diagnostischen Wert. Alles wird auf demselben Level protokolliert, Einträgen fehlen die Identifikatoren, die zur Korrelation nötig sind, dasselbe Ereignis erscheint dreimal mit unterschiedlicher Formulierung, und genau das eine, was zur Diagnose des aktuellen Vorfalls gebraucht wird, wurde nie protokolliert. Richtlinien kosten fast nichts und verwandeln Logs von einem Speicherposten in das primäre Instrument, um ein System zu verstehen, das niemand vollständig versteht.

## How to Apply ◆

> In einem Legacy-System sind Logs häufig die einzige Observability, die existiert, was ihre Qualität zum begrenzenden Faktor jeder Diagnose macht.

- **Definieren Sie, was jedes Level bedeutet**, in Begriffen dessen, wer darauf reagiert: Error bedeutet, dass jemand eingreifen muss und etwas kaputt ist, Warning bedeutet, dass etwas Unerwartetes geschah, das das System handhabte, Info erfasst bedeutsame Zustandsübergänge, und Debug ist nur für die Entwicklung. Ohne erklärte Definitionen wird alles zu Error oder alles zu Info, und beides ist äquivalent zu gar keinen Levels.
- **Verlangen Sie einen Korrelationsidentifikator** in jedem Eintrag, propagiert über Dienst- und Thread-Grenzen hinweg. Ohne ihn sind Logs eine chronologische Mischung unzusammenhängender Arbeit, und den Pfad einer Anfrage durch das System zu rekonstruieren ist der mit Abstand häufigste diagnostische Bedarf.
- **Protokollieren Sie strukturierte Daten, keine Sätze.** Schlüssel-Wert- oder JSON-Einträge können gefiltert und aggregiert werden; Prosa-Einträge können nur mit dem Auge gelesen werden, was über einen Vorfall hinaus nicht skaliert. Diese eine Änderung tut meist mehr für den diagnostischen Wert als jede Menge zusätzlichen Loggings.
- **Erklären Sie, was nie protokolliert werden darf**: Zugangsdaten, Tokens, personenbezogene Daten über das Nötige hinaus, Zahlungsdetails und vollständige Anfragekörper für alles Sensible. Log-Inhalt wird in Aggregationssysteme, Backups und Support-Tickets kopiert, und er ist eine routinemäßige Quelle von Datenexposition.
- **Neutralisieren Sie nicht vertrauenswürdige Eingabe, bevor Sie sie protokollieren.** Werte, die aus Nutzereingabe ins Log gelangen, können Zeilenumbrüche injizieren und Einträge fälschen oder den Log-Viewer ausnutzen. Das Escapen protokollierter Werte ist günstig und verhindert eine Schwachstellenklasse, die leicht übersehen wird.
- **Protokollieren Sie den Kontext, der zum Handeln nötig ist, nicht die Tatsache eines Vorkommnisses.** „Zahlungsvalidierung fehlgeschlagen" ist nicht handlungsleitend; derselbe Eintrag mit der Regel, die ablehnte, der Eingabeklasse und dem Korrelationsidentifikator ist es. Die meiste Legacy-Protokollierung ist umfangreich und kontextlos.
- **Setzen Sie Aufbewahrung nach Level und Wert**, mit erklärten Kosten. Ein Jahr aufbewahrte Debug-Einträge sind eine Speicherrechnung ohne Zweck; nach einer Woche gelöschte Error-Einträge machen Trendanalyse unmöglich.
- **Machen Sie Levels zur Laufzeit konfigurierbar**, ohne Deployment. Die Fähigkeit, die Ausführlichkeit für eine Komponente während eines Vorfalls zu erhöhen und danach wieder zu senken, ist es, was ein niedriges Standardvolumen erlaubt, was Logs nutzbar hält.
- **Bereinigen Sie beim Review.** Fragen Sie, ob jeder neue Log-Eintrag von irgendjemandem gelesen würde, und löschen Sie Einträge, die nie nützlich waren. Logging ist der einzige Code, der nie entfernt wird, weil das Entfernen riskant erscheint und das Hinzufügen verantwortungsvoll.
- **Beachten Sie die Performance-Kosten.** Synchrones Logging in einem heißen Pfad ist ein echter Engpass, und übermäßiges Logging kann mehr Ressourcen verbrauchen als die protokollierte Arbeit selbst.

## Tradeoffs ⇄

> Richtlinien machen Logs nutzbar und günstiger, aber das Nachrüsten über eine alte Codebasis hinweg ist erhebliche Arbeit, und die Reduzierung des Volumens riskiert immer, etwas zu entfernen, das wichtig gewesen wäre.

**Vorteile:**

- Diagnose wird wesentlich schneller, da korrelierte strukturierte Einträge erlauben, den Pfad einer Anfrage zu rekonstruieren statt zu erschließen.
- Volumen und Speicherkosten fallen, oft dramatisch, weil das meiste Volumen in einem unverwalteten System keine Information trägt.
- Alarmierung wird möglich. Ein Error-Level, das echt bedeutet, dass etwas kaputt ist, kann alarmiert werden; eines, das Tausende Male am Tag feuert, kann es nicht.
- Die Exposition sensibler Daten durch Logs wird verhindert, was einen Kanal schließt, der leicht übersehen wird, gerade weil Logs nicht als Datenspeicher gedacht werden.
- Log Injection wird geschlossen, eine Schwachstellenklasse, die günstig zu verhindern und im Nachhinein umständlich zu erkennen ist.

**Kosten und Risiken:**

- Richtlinien über eine große Legacy-Codebasis nachzurüsten ist eine große mechanische Aufgabe ohne sichtbare Ausgabe, und sie wird selten abgeschlossen.
- Die Reduzierung des Volumens kann einen Eintrag entfernen, der bei einem zukünftigen Vorfall entscheidend gewesen wäre, und dieser Verlust wird nur später entdeckt.
- Strukturiertes Logging erfordert Framework-Unterstützung und manchmal eine Migration, was in älteren Stacks echter Aufwand ist.
- Zur Laufzeit konfigurierbare Levels fügen Konfigurationsfläche hinzu, und eine Fehlkonfiguration kann Logging vollständig stumm schalten — ein Fehler, der unsichtbar ist, bis etwas schiefgeht.
- Inkonsistent angewendete Richtlinien produzieren eine Codebasis mit zwei Logging-Konventionen, was für Filter- und Aggregationszwecke schlimmer sein kann als eine schlechte Konvention, einheitlich angewendet.

## How It Could Be

Ein Team, das eine Zahlungsplattform pflegte, erzeugte etwa 400 Gigabyte Logs pro Tag, zu erheblichen Kosten, und konnte trotzdem routinemäßig Ausfälle nicht diagnostizieren. Die Untersuchung fand, dass ein Eintrag — geschrieben auf Error-Level bei jedem wiederholbaren Timeout, was normal vorkam — etwa 60 Prozent des Volumens ausmachte. Nichts trug einen Korrelationsidentifikator, sodass die Rekonstruktion einer einzigen fehlgeschlagenen Zahlung bedeutete, nach Zeitstempel über vier Dienste zu greppen und zu raten. Sie schrieben eine zweiseitige Richtlinie: vier definierte Levels, ein verpflichtender Korrelationsidentifikator, strukturierte Einträge und eine Liste verbotener Inhalte. Das Nachrüsten des Korrelationsidentifikators dauerte drei Wochen. Das Volumen fiel auf etwa 40 Gigabyte pro Tag, und die mediane Zeit zur Diagnose eines Zahlungsfehlschlags fiel von über zwei Stunden auf etwa fünfzehn Minuten — fast vollständig, weil ein einzelner Filter nun den vollständigen Pfad einer Zahlung zurückgab.

Die Liste verbotener Inhalte fand etwas, wonach das Team nicht gesucht hatte. Eine Suche des Log-Aggregationssystems nach den neu verbotenen Mustern förderte einen Debug-Eintrag zutage, vier Jahre zuvor während eines Integrationsproblems hinzugefügt und nie entfernt, der vollständige eingehende API-Anfragen einschließlich Authentifizierungsheader protokollierte. Diese Logs wurden 90 Tage aufbewahrt und waren für jeden mit Zugriff auf das Aggregationswerkzeug lesbar, was der Großteil der Engineering-Organisation war. Der Eintrag wurde entfernt und die aufbewahrten Logs bereinigt, und der Vorfall wurde zum Grund, warum die Organisation einen wiederkehrenden automatisierten Scan des Log-Inhalts nach Zugangsdatenmustern hinzufügte.
