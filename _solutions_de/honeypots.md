---
title: Honeypots
description: Einsatz speziell abgesicherter Systeme als Köder für Angreifer.
category:
- Security
problems:
- monitoring-gaps
- authentication-bypass-vulnerabilities
- slow-incident-resolution
- data-protection-risk
- insufficient-audit-logging
layout: solution
lang: de
en_slug: honeypots
related_solutions:
- slug: endpoint-detection-and-response
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.8
- slug: threat-intelligence
  similarity: 0.8
- slug: logging-and-monitoring
  similarity: 0.8
- slug: incident-response-measures
  similarity: 0.8
- slug: network-segmentation
  similarity: 0.75
---

## Description

Ein Honeypot ist ein Köder-System, eine Köder-Zugangsdaten oder eine Köder-Datei, bewusst platziert, um für einen Angreifer wertvoll auszusehen, während sie keine echten Daten oder Funktion enthält, sodass jede Interaktion damit von Natur aus verdächtig ist — kein legitimer Nutzer oder Prozess hat irgendeinen Grund, sie anzufassen. Dies gibt Honeypots eine ungewöhnlich niedrige Falsch-Positiv-Rate im Vergleich zu den meisten Erkennungsmechanismen: Statt bösartigen Verkehr statistisch von einem großen Volumen legitimer Aktivität zu unterscheiden, erfordert ein Honeypot-Alarm kein solches Urteil, weil seine bloße Existenz die Falle ist. Diese Eigenschaft ist besonders wertvoll rund um Legacy-Systeme, die häufige Angriffsziele sind, gerade wegen ihrer bekannten, oft ungepatchten Schwachstellen, und die typischerweise schwächeres natives Logging und Monitoring haben als moderne Komponenten — ein Honeypot kann diese Lücke kompensieren, indem er ein hochzuverlässiges Signal erzeugt, ohne jegliche Modifikation am Legacy-System selbst zu erfordern. Honeytoken-Zugangsdaten, eingebettet in Konfigurationsdateien oder Repositories, dienen demselben Zweck in kleinerem Maßstab: Sie erlauben einem Team, Zugangsdatendiebstahl oder laterale Bewegung zu erkennen, ohne die Störung, echte, tief eingebettete Legacy-Geheimnisse sofort zu rotieren. Das Hauptbetriebsrisiko ist, dass ein schlecht isolierter Honeypot selbst zu einem Sprungbrett in echte Systeme werden kann, sodass Honeypots ausreichend eingedämmt sein müssen, um netzwerkverbunden zu erscheinen, während sie von allem tatsächlich Wertvollem abgeschottet bleiben, und sie müssen periodisch erneuert werden, damit ausgeklügelte Angreifer nicht einfach lernen können, sie zu erkennen und zu meiden.

## How to Apply ◆

> Legacy-Systeme sind attraktive Ziele für Angreifer wegen ihrer bekannten Schwachstellen und oft schwachen Monitorings. Honeypots ergänzen bestehende Sicherheitskontrollen, indem Köder-Systeme eingesetzt werden, die Angreifer anlocken und erkennen und frühe Warnung und Intelligenz über Angriffsmethoden liefern.

- Setzen Sie Low-Interaction-Honeypots ein, die die externen Schnittstellen des Legacy-Systems emulieren (Login-Seiten, API-Endpunkte, Datenbank-Ports), aber keine echten Daten enthalten. Diese sind schnell einzurichten und erkennen automatisiertes Scanning und opportunistische Angriffe.
- Platzieren Sie Honeypot-Endpunkte innerhalb des Netzwerksegments des Legacy-Systems, um laterale Bewegung zu erkennen. Interne Honeypots, die nie legitimen Verkehr erhalten sollten, liefern hochzuverlässige Alarme — jede Verbindung zu ihnen deutet auf unautorisierte Aktivität oder Kompromittierung hin.
- Erstellen Sie Honeytoken-Zugangsdaten (gefälschte Datenbankkonten, API-Schlüssel, Dienstzugangsdaten), eingebettet an Orten, die Angreifer häufig durchsuchen: Konfigurationsdateien, Quellcode-Repositories und gemeinsam genutzte Netzlaufwerke. Jede Nutzung dieser Zugangsdaten löst einen sofortigen Alarm aus.
- Setzen Sie Honeypot-Dateien ein (gefälschte Kundendatenbanken, Dummy-Konfigurationsdateien mit attraktiven Namen) an gemeinsam genutzten Speicherorten. Zugriff auf diese Dateien, die kein legitimer Nutzer oder Prozess anfassen sollte, deutet entweder auf eine Insider-Bedrohung oder einen Angreifer mit Systemzugriff hin.
- Konfigurieren Sie detailliertes Logging auf allen Honeypot-Komponenten, um Angreifertechniken, -werkzeuge und -ziele zu erfassen. Diese Intelligenz verbessert die Verteidigung des echten Legacy-Systems, indem offenbart wird, worauf Angreifer abzielen.
- Stellen Sie sicher, dass Honeypots ausreichend isoliert sind, damit ein Angreifer, der den Honeypot kompromittiert, nicht zu echten Systemen vordringen kann. Honeypots sollten mit dem Netzwerk verbunden erscheinen, aber innerhalb einer überwachten Sandbox eingedämmt sein.

## Tradeoffs ⇄

> Honeypots bieten frühe Angriffserkennung und Bedrohungsintelligenz mit niedrigen Falsch-Positiv-Raten, müssen aber realistisch gepflegt und angemessen isoliert werden.

**Vorteile:**

- Erkennt Angriffe, die andere Sicherheitskontrollen umgehen, indem hochzuverlässige Alarme geliefert werden — jede Interaktion mit einem Honeypot ist per Definition verdächtig.
- Liefert Intelligenz über Angreiferwerkzeuge, -techniken und -ziele, die die Verteidigung des echten Legacy-Systems verbessert.
- Lenkt Angreiferaufmerksamkeit und -aufwand auf wertlose Ziele um und verschafft Zeit für Erkennung und Reaktion.
- Niedrige Falsch-Positiv-Rate, da kein legitimer Nutzer oder System mit Honeypot-Ressourcen interagieren sollte.

**Kosten und Risiken:**

- Ein kompromittierter, nicht angemessen isolierter Honeypot kann als Sprungbrett genutzt werden, um echte Systeme anzugreifen.
- Honeypots erfordern laufende Pflege, um realistisch zu bleiben — veraltete oder offensichtlich gefälschte Honeypots werden von ausgeklügelten Angreifern leicht identifiziert und ignoriert.
- Das Einsetzen von Honeypots führt zusätzliche Systeme ein, die überwacht, gepatcht (oder bewusst kontrolliert ungepatcht gelassen) und verwaltet werden müssen.
- Rechtliche und ethische Überlegungen können bei der Aufzeichnung von Angreiferaktivität gelten, abhängig von der Jurisdiktion.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Honeypots Bedrohungen erkennen, die auf Legacy-Systeme abzielen.

Ein Unternehmen betreibt einen Legacy-Datenbankserver, der von SQL-Injection-Angriffen anvisiert wurde. Sie setzen einen Honeypot ein, der die Login-Schnittstelle der Legacy-Datenbank auf einer nahegelegenen IP-Adresse mit einem absichtlich schwachen Passwort emuliert. Innerhalb einer Woche protokolliert der Honeypot 14 Verbindungsversuche von drei verschiedenen IP-Adressen mit automatisierten Credential-Brute-Forcing-Werkzeugen. Die erfassten Angriffsmuster offenbaren, dass die Angreifer ein spezifisches Exploit-Toolkit nutzen, das auf die Legacy-Datenbankversion abzielt. Diese Intelligenz erlaubt dem Sicherheitsteam, ihre Intrusion-Detection-Signaturen und Firewall-Regeln zu aktualisieren, um diese spezifischen Angriffsmuster auf dem echten Datenbankserver zu blockieren, und verhindert Angriffe, die vom bestehenden Monitoring nicht erkannt worden wären.

Ein Legacy-Quellcode-Repository enthält fest codierte Datenbankzugangsdaten in Konfigurationsdateien. Statt diese Zugangsdaten sofort zu rotieren (was die Koordination von Änderungen über mehrere Legacy-Komponenten hinweg erfordern würde), erstellt das Sicherheitsteam Honeytoken-Datenbankzugangsdaten in denselben Konfigurationsdateien neben den echten. Die Honeytokens werden überwacht — jeder Authentifizierungsversuch mit ihnen löst einen Alarm aus. Drei Wochen nach dem Einsatz feuert ein Alarm: Jemand versucht, sich mit den Honeytoken-Zugangsdaten bei der Datenbank zu authentifizieren. Die Untersuchung offenbart, dass der Laptop eines Auftragnehmers kompromittiert wurde und der Angreifer Zugangsdaten aus einem geklonten Repository extrahierte. Der Honeytoken-Alarm liefert 4 Stunden Vorwarnung, bevor der Angreifer die echten Zugangsdaten versucht, die das Team in diesem Fenster rotiert.
