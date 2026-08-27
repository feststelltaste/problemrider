---
title: Serialisierungsoptimierung
description: Wahl effizienter Serialisierungsformate für performance-kritischen
  Datenaustausch.
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/serialization-optimization/
problems:
- serialization-deserialization-bottlenecks
- high-api-latency
- microservice-communication-overhead
- network-latency
- external-service-delays
- excessive-object-allocation
- garbage-collection-pressure
- inefficient-code
- algorithmic-complexity-problems
- resource-contention
layout: solution
lang: de
en_slug: serialization-optimization
related_solutions:
- slug: efficient-algorithms
  similarity: 0.8
- slug: caching-strategy
  similarity: 0.75
- slug: profiling
  similarity: 0.7
- slug: resource-usage-optimization
  similarity: 0.7
- slug: api-calls-optimization
  similarity: 0.7
- slug: compression
  similarity: 0.7
---

## Description

Serialisierungsoptimierung ersetzt ein umständliches, reflection-basiertes Serialisierungsformat durch ein kompakteres oder effizienteres — binäre Formate wie Protocol Buffers für Service-zu-Service-Aufrufe, selektive Feldeinbeziehung für clientseitiges JSON —, wodurch die CPU-, Speicher- und Netzwerkkosten der Umwandlung von Daten zu und von Wire-Format reduziert werden. Legacy-Systeme greifen häufig standardmäßig auf welches Format auch immer zurück, das vor Jahren aus Gründen menschlicher Lesbarkeit oder historischer Bequemlichkeit gewählt wurde, und da Service-zu-Service-Aufrufvolumen und Payload-Größe seither gewachsen sind, kann diese Wahl am Ende einen wirklich großen Anteil der Gesamt-Request-Zeit verbrauchen — 20 bis 40 Prozent sind üblich —, völlig unsichtbar, bis jemand speziell die Serialisierungsschicht profiliert statt der umgebenden Geschäftslogik. Der Gewinn ist erheblich und erfordert überhaupt kein Anfassen der Geschäftslogik, aber er tauscht menschliche Lesbarkeit für Debugging ein und erfordert eine sorgfältige Übergangsphase, wenn die Formatänderung auf ein bereits in Produktion laufendes System ausgerollt werden muss.

## How to Apply ◆

> Legacy-Systeme nutzen häufig umständliche Serialisierungsformate, die aus Gründen menschlicher Lesbarkeit oder historischer Gründe gewählt wurden, statt aus Performance-Gründen. Während Datenvolumina und Servicekommunikationsfrequenz wachsen, wird Serialisierungs-Overhead zu einem erheblichen Anteil der Gesamtverarbeitungszeit. Die Optimierung der Serialisierung reduziert Latenz, Speicherverbrauch und Netzwerkbandbreitennutzung, ohne Änderungen an der Geschäftslogik zu erfordern.

- Messen Sie Serialisierungs-Overhead vor der Optimierung. Profilieren Sie die API-Anfragenverarbeitung, um zu bestimmen, welcher Prozentsatz der Gesamtantwortzeit für Serialisierung und Deserialisierung aufgewendet wird. In Legacy-Microservice-Architekturen kann Serialisierung 20-40 % der Anfragenverarbeitungszeit ausmachen — aber dies ist nur mit gezieltem Profiling der Serialisierungsschicht sichtbar.
- Ersetzen Sie umständliche Textformate durch kompakte binäre Formate für Service-zu-Service-Kommunikation, wo menschliche Lesbarkeit nicht erforderlich ist. Protocol Buffers, FlatBuffers, MessagePack oder Avro erreichen typischerweise 3-10-fach kleinere Payload-Größen und 5-20-fach schnellere Serialisierung im Vergleich zu JSON oder XML, mit dem zusätzlichen Vorteil der Schema-Durchsetzung.
- Wechseln Sie für JSON-basierte APIs, die aus Client-Kompatibilitätsgründen JSON bleiben müssen, zu hochperformanten Serialisierungsbibliotheken. Ersetzen Sie reflection-basierte Serialisierer durch codegenerierte oder compile-time Serialisierer (System.Text.Json statt Newtonsoft.Json in .NET, Jackson mit Compile-Time-Modulen in Java, orjson oder ujson in Python), die den Laufzeit-Overhead von Reflection vermeiden.
- Implementieren Sie selektive Serialisierung, um das Marshalling von Daten zu vermeiden, die Konsumenten nicht benötigen. Statt ganze Objektgraphen zu serialisieren, definieren Sie Antwortprojektionen oder Feldauswahl (ähnlich der GraphQL-Feldauswahl), die nur die vom Aufrufer benötigten Felder einbeziehen. Dies ist besonders wirkungsvoll, wenn Legacy-APIs tief verschachtelte oder übermäßig breite Antwortobjekte zurückgeben.
- Nutzen Sie Streaming-Serialisierung für große Payloads, statt die gesamte serialisierte Ausgabe im Speicher zu konstruieren, bevor sie gesendet wird. Streamen Sie JSON-Arrays, CSV-Zeilen oder Binärdatensätze direkt in den Ausgabestream, sobald sie produziert werden, was den Spitzenspeicherverbrauch und die Time-to-First-Byte-Latenz reduziert.
- Vermeiden Sie unnötige Umwege durch Serialisierung. In Legacy-Systemen werden Daten manchmal zu einem String serialisiert, gespeichert, dann deserialisiert und in einem anderen Format re-serialisiert — jeder Schritt verbraucht CPU und Speicher. Identifizieren Sie diese Ketten und beseitigen Sie zwischengeschaltete Serialisierungsschritte, indem Sie native Objekte oder binäre Repräsentationen durch Verarbeitungspipelines leiten.
- Pre-serialisieren und cachen Sie Antworten für häufig angefragte, sich langsam ändernde Daten. Wenn dieselbe API-Antwort an viele Clients ausgeliefert wird, serialisieren Sie sie einmal und cachen Sie die serialisierten Bytes, statt bei jeder Anfrage aus Objekten neu zu serialisieren. Dies kombiniert Caching-Vorteile mit Serialisierungsoptimierung.
- Wählen Sie Serialisierungsformate, die Schema-Evolution für APIs unterstützen, die sich mit der Zeit ändern. Protocol Buffers und Avro unterstützen das Hinzufügen und Entfernen von Feldern, ohne bestehende Konsumenten zu brechen, was kritisch ist in Legacy-Umgebungen, in denen koordinierte Deployments über alle Services hinweg schwierig sind.
- Komprimieren Sie serialisierte Payloads für die Netzwerkübertragung mit gzip oder zstd, wenn die Payload-Größe die Kompressions-Overhead-Schwelle überschreitet (typischerweise über 1KB). Aktivieren Sie HTTP-Kompression auf Webserver- oder API-Gateway-Ebene für textbasierte Formate, und bewerten Sie, ob binäre Formate von zusätzlicher Kompression profitieren.

## Tradeoffs ⇄

> Serialisierungsoptimierung reduziert Latenz und Ressourcenverbrauch für Datenaustauschoperationen, kann aber menschliche Lesbarkeit opfern und Komplexität der Formatmigration einführen.

**Vorteile:**

- Reduziert API-Antwortzeiten durch Beseitigung von Serialisierungs-Overhead, der die Anfragenverarbeitung in Legacy-Systemen mit großen oder tief verschachtelten Antwortobjekten dominieren kann.
- Verringert den Netzwerkbandbreitenverbrauch durch kleinere Payload-Größen, was besonders wirkungsvoll für hochfrequente Service-zu-Service-Kommunikation in Microservice-Architekturen ist.
- Reduziert Garbage-Collection-Druck durch Minimierung temporärer Objektallokation während Serialisierung und Deserialisierung, was den Gesamtdurchsatz der Anwendung verbessert.
- Senkt die CPU-Auslastung durch Ersetzung reflection-basierter Serialisierung durch codegenerierte oder binäre Alternativen, was Verarbeitungskapazität für Geschäftslogik freisetzt.
- Verbessert Time-to-First-Byte durch Streaming-Serialisierung, was es Clients ermöglicht, mit der Verarbeitung partieller Antworten zu beginnen, bevor die gesamte Payload zusammengesetzt ist.

**Kosten und Risiken:**

- Binäre Serialisierungsformate opfern menschliche Lesbarkeit, was Debugging und Fehlersuche erschwert. Teams brauchen Werkzeuge, um binäre Payloads zu inspizieren, und Logging muss angepasst werden, um dekodierte Repräsentationen aufzuzeichnen.
- Die Migration von einem Serialisierungsformat zu einem anderen in einem laufenden System erfordert eine Übergangsphase, in der beide Formate unterstützt werden, was Komplexität und das Risiko von Kompatibilitätsproblemen erhöht.
- Schema-basierte Formate (Protocol Buffers, Avro) erfordern Schema-Verwaltungstooling und -prozesse, die im Legacy-Entwicklungsworkflow möglicherweise nicht existieren.
- Kompression fügt CPU-Overhead hinzu, der für kleine Payloads oder auf CPU-beschränkten Systemen möglicherweise nicht gerechtfertigt ist; der Break-Even-Punkt zwischen Kompressionskosten und Netzwerkeinsparungen muss für jeden Anwendungsfall gemessen werden.
- Das Ändern von Serialisierungsbibliotheken in Legacy-Code kann subtile Verhaltensunterschiede einführen, wie Nullwerte, Daten, numerische Präzision und Zeichenkodierung behandelt werden, was gründliches Kompatibilitätstesting erfordert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Serialisierungsoptimierung Performance-Probleme in Legacy-Systemen adressiert.

Die Sendungsverfolgungs-API eines Logistikunternehmens gab Sendungsdetails einschließlich vollständiger Routenhistorie, Fahrerinformationen und Paketabmessungen als tief verschachteltes XML-Dokument zurück. Die Serialisierung der Antwort für eine einzelne Sendung mit 50 Routenpunkten dauerte 180ms und produzierte eine 95KB-Payload. Die API bediente 500 Anfragen pro Sekunde, und XML-Serialisierung verbrauchte 35 % der Server-CPU-Kapazität. Das Team implementierte eine zweiphasige Optimierung: Für die öffentliche REST-API, die von mobilen Clients konsumiert wird, wechselten sie von XML zu JSON mit selektiver Feldeinbeziehung (nur die tatsächlich von jedem Client genutzten Felder zurückgeben), was die Payload-Größe auf 12KB und die Serialisierungszeit auf 15ms reduzierte. Für interne Service-zu-Service-Aufrufe zwischen dem Tracking-Service und dem Benachrichtigungsservice übernahmen sie Protocol Buffers, was den Serialisierungs-Overhead auf 2ms und die Payload-Größe auf 3KB reduzierte. Der gesamte CPU-Verbrauch durch Serialisierung sank von 35 % auf 4 %, was Kapazität freisetzte, um das dreifache vorherige Anfragenvolumen auf derselben Hardware zu bedienen.

Eine Gesundheitsinteroperabilitätsplattform tauschte HL7-FHIR-Ressourcen zwischen Krankenhaussystemen mit JSON und dem Standard-Jackson-Serialisierer aus. Jedes Patienten-Bundle enthielt 200-500 Ressourcen mit tief verschachtelten Strukturen, und die Serialisierung eines einzelnen Bundles dauerte 800ms mit 400MB temporärer Objektallokation aufgrund von Jacksons reflection-basiertem Feldzugriff. Das Team wechselte zu Jackson mit Compile-Time-Codegenerierungsmodulen und implementierte einen Streaming-Serialisierer, der Ressourcen direkt in den HTTP-Antwortstream schrieb, statt die gesamte Antwort im Speicher aufzubauen. Die Serialisierungszeit sank auf 120ms, die Speicherallokation verringerte sich um 85 %, und die GC-Pausenhäufigkeit sank von alle 10 Sekunden auf alle 2 Minuten. Der Streaming-Ansatz verbesserte auch die Client-Erfahrung, weil das empfangende System mit der Verarbeitung von Ressourcen beginnen konnte, während das Bundle noch übertragen wurde.

Ein Finanzdatenaggregationsdienst sammelte Marktdaten von 15 externen Anbietern und verteilte sie an 200 interne Konsumenten. Die Legacy-Implementierung empfing Daten im nativen Format jedes Anbieters (eine Mischung aus CSV, XML und proprietärem Binärformat), deserialisierte sie in Java-Objekte und re-serialisierte sie dann zu JSON für jede Konsumentenverbindung. Während der Handelszeiten verbrauchte dieser dreifache Serialisierungszyklus 60 % der CPU-Kapazität und erzeugte 12 Millionen temporäre Objekte pro Minute, was konstanten GC-Druck und 200ms-Latenzspitzen alle paar Sekunden verursachte. Das Team gestaltete die Pipeline neu, um alle eingehenden Daten bei Empfang einmal zu Avro-Format zu normalisieren, die serialisierten Avro-Bytes zu cachen und Konsumenten direkt aus dem gecachten Binärformat zu bedienen. Konsumenten, die JSON benötigten, erhielten eine einzige Avro-zu-JSON-Übersetzung am Edge. Der CPU-Verbrauch sank auf 15 %, GC-Pausen wurden vernachlässigbar, und die End-to-End-Datenverteilungslatenz sank von durchschnittlich 800ms auf 50ms.
