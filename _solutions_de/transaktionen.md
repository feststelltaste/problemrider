---
title: Transaktionen
description: Gruppierung mehrerer Operationen in eine atomare, konsistente
  Einheit.
category:
- Architecture
- Database
problems:
- silent-data-corruption
- race-conditions
- data-migration-integrity-issues
- inconsistent-behavior
- long-running-database-transactions
- long-running-transactions
- deadlock-conditions
- cascade-failures
- synchronization-problems
- lock-contention
layout: solution
lang: de
en_slug: transactions
related_solutions:
- slug: concurrency-control
  similarity: 0.8
- slug: write-ahead-logging
  similarity: 0.75
- slug: saga-pattern
  similarity: 0.75
- slug: batch-processing
  similarity: 0.75
- slug: idempotency-design
  similarity: 0.75
- slug: data-integrity
  similarity: 0.75
---

## Description

Eine Transaktion gruppiert eine Reihe verwandter Datenmodifikationen in eine einzige atomare Einheit, die entweder vollständig committet oder vollständig zurückgerollt wird, sodass ein Fehlschlag mitten in einer Operation das System nie in einem inkonsistenten, halb aktualisierten Zustand zurücklässt. Diese Garantie wird von der Datenbank durch Isolationsstufen und Rollback-Mechanismen bereitgestellt, aber sie greift nur dort, wo die Anwendung Transaktionsgrenzen explizit abgrenzt — und viele Legacy-Systeme taten dies nie, da sie geschrieben wurden, um sich auf Auto-Commit-Modus zu verlassen, bei dem jede Anweisung ihre eigene implizite Transaktion ist, ohne Atomizität über die Sequenz von Anweisungen hinweg, die zusammen eine echte Geschäftsoperation darstellen. Diese Lücke ist eine häufige Quelle genau der Art stiller Datenbeschädigung, die alternde Systeme plagt: ein Auftrag ohne entsprechende Bestandsreduzierung erfasst, eine Belastung ohne passende Gutschrift gebucht, erst viel später als eine Diskrepanz entdeckt, die niemand erklären kann. Solche mehrstufigen Operationen in ordentliche Transaktionen zu verpacken, Isolationsstufen nicht strenger als nötig zu wählen, und das Konzept auf Sagas mit kompensierenden Aktionen zu erweitern, wo eine einzelne ACID-Transaktion nicht mehrere Dienste umspannen kann, stellt die Alles-oder-nichts-Garantie wieder her, die der ursprünglichen Implementierung fehlte. In der Legacy-Modernisierung sind Transaktionen häufig eine der ersten Korrektheitsfixes, die angewendet werden, weil sie eine Defektklasse schließen, die sonst unbegrenzt neue, schwer reproduzierbare Datenintegritätsvorfälle erzeugt.

## How to Apply ◆

> Legacy-Systeme führen oft mehrstufige Datenmodifikationen ohne transaktionale Garantien durch, was Daten in inkonsistenten Zuständen zurücklässt, wenn Fehlschläge mitten in einer Operation auftreten. Ordentliches Transaktionsmanagement stellt sicher, dass verwandte Operationen entweder alle als Einheit gelingen oder alle fehlschlagen.

- Identifizieren Sie alle mehrstufigen Operationen im Legacy-System, bei denen teilweiser Abschluss Daten in einem inkonsistenten Zustand zurücklassen würde. Häufige Beispiele sind Auftragsverarbeitung (Bestand reservieren, Zahlung belasten, Versand erstellen), Finanzüberweisungen (ein Konto belasten, ein anderes gutschreiben) und Stammdaten-Updates, die mehrere Tabellen umspannen.
- Verpacken Sie verwandte Datenbankoperationen in explizite Transaktionen mit angemessenen Isolationsstufen. Viele Legacy-Systeme verlassen sich auf Auto-Commit-Modus, bei dem jede SQL-Anweisung ihre eigene Transaktion ist, was keine Atomizität über verwandte Operationen bietet.
- Wählen Sie die minimale Isolationsstufe, die Korrektheit für jeden Anwendungsfall bietet. READ COMMITTED ist für die meisten Operationen ausreichend; SERIALIZABLE verhindert alle Anomalien, reduziert aber die Nebenläufigkeit dramatisch. Legacy-Systeme mit hoher Konkurrenz leiden überproportional unter übermäßig restriktiven Isolationsstufen.
- Implementieren Sie das Saga-Pattern für Operationen, die mehrere Dienste oder Datenbanken umspannen, wo eine einzelne ACID-Transaktion nicht möglich ist. Definieren Sie kompensierende Transaktionen für jeden Schritt, sodass die Gesamtoperation zurückgerollt werden kann, wenn ein Schritt fehlschlägt.
- Halten Sie den Transaktionsumfang so klein wie möglich — erwerben Sie Locks spät, geben Sie sie früh frei. Legacy-Systeme halten häufig Transaktionen während Nutzerinteraktionen oder externer API-Aufrufe offen, was Lock-Konkurrenz und Deadlocks verursacht.
- Fügen Sie Idempotenzschlüssel zu transaktionsauslösenden Operationen hinzu, sodass Retries nach Timeout oder Netzwerkfehler nicht zu doppelter Verarbeitung führen. Dies ist kritisch für Legacy-Systeme, bei denen der Aufrufer nicht zuverlässig bestimmen kann, ob eine ausgelaufene Transaktion committet oder zurückgerollt wurde.
- Implementieren Sie ordentliches Fehlerhandling, das Transaktionen bei jeder Exception zurückrollt, einschließlich unerwarteter Laufzeitfehler. Legacy-Code fängt und verschluckt oft Exceptions, ohne zurückzurollen, und lässt teilweise committete Daten zurück.

## Tradeoffs ⇄

> Transaktionen bieten Datenkonsistenzgarantien, die Beschädigung durch Teiloperationen verhindern, führen aber Konkurrenz-Overhead und Komplexität in verteilten Systemen ein.

**Vorteile:**

- Verhindern Datenbeschädigung durch teilweisen Operationsabschluss, indem sie Alles-oder-nichts-Semantik für verwandte Änderungen sicherstellen.
- Vereinfachen Fehlerwiederherstellung, indem sie automatischen Rollback bieten, wenn ein Schritt in einer mehrstufigen Operation fehlschlägt.
- Ermöglichen nebenläufigen Zugriff auf gemeinsam genutzte Daten mit wohldefinierten Konsistenzgarantien durch Isolationsstufen.
- Bieten eine Grundlage für Audit-Trails und Compliance, indem sichergestellt wird, dass protokollierte Zustandsübergänge immer vollständig und konsistent sind.

**Kosten und Risiken:**

- Langlaufende Transaktionen halten Locks, die andere Operationen blockieren, was den Systemdurchsatz reduziert und Konkurrenzengpässe in Legacy-Systemen mit gemeinsam genutzten Datenbanken schafft.
- Verteilte Transaktionen über mehrere Datenbanken oder Dienste hinweg sind komplex, brüchig und reduzieren die Verfügbarkeit erheblich, was oft saga-basierte Alternativen erfordert, die schwerer korrekt zu implementieren sind.
- Deadlocks werden wahrscheinlicher, während der Transaktionsumfang zunimmt, was Erkennung, Timeout und Retry-Logik erfordert.
- Legacy-Datenbanken könnten begrenzte Transaktionsunterstützung oder unerwartetes Verhalten unter bestimmten Isolationsstufen haben, was sorgfältiges Testen erfordert.
- Transaktions-Retries nach Fehlschlägen können doppelte Nebenwirkungen verursachen (E-Mails senden, externe APIs aufrufen), es sei denn, Idempotenz ist explizit eingebaut.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie ordentliches Transaktionsmanagement Datenbeschädigung in Legacy-Systemen verhindert.

Ein Legacy-E-Commerce-System verarbeitet Aufträge durch Ausführung einer Sequenz von fünf SQL-Anweisungen: den Auftragskopf einfügen, Auftragspositionen einfügen, Bestand reduzieren, einen Zahlungsdatensatz einfügen und die Auftragsanzahl des Kunden aktualisieren. Diese Anweisungen werden einzeln mit aktiviertem Auto-Commit ausgeführt. Wenn die Datenbankverbindung abbricht, nachdem der Zahlungsdatensatz eingefügt wurde, aber bevor der Bestand reduziert wurde, erstellt das System einen Auftrag mit einer Zahlung, aber ohne reservierten Bestand. Dieselben Artikel werden dann an einen anderen Kunden verkauft, was zu Überverkauf führt. Das Team verpackt alle fünf Anweisungen in eine einzige Datenbanktransaktion und stellt sicher, dass entweder alle gelingen oder keine. Sie fügen auch eine eindeutige Auftrags-ID als Idempotenzschlüssel hinzu, sodass, wenn der Client nach einem Verbindungsfehler erneut versucht, der Retry die bestehende Transaktion erkennt und das Ergebnis zurückgibt, statt einen doppelten Auftrag zu erstellen.

Ein Legacy-Bankensystem überweist Gelder zwischen Konten mit zwei separaten UPDATE-Anweisungen — eine, um das Quellkonto zu belasten, und eine, um das Zielkonto gutzuschreiben. Unter hoher Last stürzt die Anwendung gelegentlich zwischen der Belastungs- und der Gutschriftoperation ab, was dazu führt, dass Geld vom Quellkonto verschwindet, ohne im Zielkonto zu erscheinen. Das Team implementiert explizite Transaktionsverpackung mit SERIALIZABLE-Isolation für Überweisungsoperationen. Sie entdecken auch, dass der Legacy-Code Datenbank-Exceptions fängt und protokolliert, aber die Transaktion nicht zurückrollt, was teilweise committete Änderungen zurücklässt. Nach der Behebung des Fehlerhandlings, um bei Fehlschlag immer zurückzurollen, und der Implementierung automatischer Retries mit exponentiellem Backoff für Serialisierungskonflikte hat das System im folgenden Jahr null Geldbetrag-Diskrepanz-Vorfälle.
