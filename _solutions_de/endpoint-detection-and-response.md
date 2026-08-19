---
title: Endpoint Detection and Response
description: Kontinuierliche Echtzeit-Überwachung von Endpunkten auf Bedrohungen.
category:
- Security
- Operations
problems:
- system-outages
- monitoring-gaps
- slow-incident-resolution
- data-protection-risk
- authentication-bypass-vulnerabilities
- constant-firefighting
layout: solution
lang: de
en_slug: endpoint-detection-and-response
related_solutions:
- slug: honeypots
  similarity: 0.8
- slug: malware-protection
  similarity: 0.8
- slug: incident-response-measures
  similarity: 0.8
- slug: security-monitoring
  similarity: 0.75
- slug: logging-and-monitoring
  similarity: 0.75
- slug: threat-intelligence
  similarity: 0.75
---

## Description

Endpoint Detection and Response setzt Agenten auf Servern und Arbeitsplatzrechnern ein, die kontinuierlich Prozessausführung, Dateiintegrität und Verhaltensmuster gegen eine etablierte Baseline überwachen, was sowohl Echtzeiterkennung von Kompromittierung als auch, wo das Vertrauen ausreichend hoch ist, automatisierte Eindämmungsaktionen wie die Isolation eines Endpunkts vom Netzwerk ermöglicht. Dies ist für Legacy-Systeme besonders folgenreich, weil sie häufig auf Endpunkten laufen — ältere Windows-Server-Versionen, dedizierte Werkshallen-Arbeitsplätze, ungepatchte Linux-Hosts —, die aus Angst, die Anwendungskompatibilität zu brechen, selten Sicherheitspatches erhalten, und folglich bekannte, ausnutzbare Schwachstellen anhäufen, die weit länger bestehen bleiben, als sie es in aktiv gepflegter Infrastruktur würden. Ohne EDR kann eine Kompromittierung auf einem solchen Endpunkt monatelang unentdeckt bleiben, da Legacy-Systemen oft jegliche eigene Sicherheitsinstrumentierung auf Anwendungsebene fehlt und sie vollständig auf der Annahme beruhen, dass nichts Schlimmes geschieht — eine Annahme, die EDR durch tatsächliche kontinuierliche Beobachtung ersetzt. Weil Legacy-Systeme häufig vorhersehbare, repetitive, wenig variierende Nutzungsmuster aufweisen, ist Verhaltens-Baselining oft ungewöhnlich effektiv darin, echt anomale Aktivität zu markieren, etwa einen unerwarteten Prozess, der von einem Service-Konto gestartet wird, das sich nie zuvor so verhalten hat. Die Tradeoffs sind konkret: EDR-Agenten verbrauchen CPU und Speicher auf Servern, die möglicherweise bereits ressourcenbeschränkt sind, die Kompatibilität mit älteren Betriebssystemen ist nicht garantiert, und automatisierte Isolationsaktionen tragen ein echtes Risiko, einen ungeplanten Ausfall zu verursachen, wenn sie fälschlich gegen einen kritischen Produktionsendpunkt ausgelöst werden.

## How to Apply ◆

> Legacy-Systeme laufen oft auf Endpunkten (Server, Arbeitsplätze, Terminals) mit minimaler oder keiner Bedrohungserkennung, was es unmöglich macht, Kompromittierung zu identifizieren, bis erheblicher Schaden entstanden ist. Endpoint Detection and Response (EDR) bietet kontinuierliche Sichtbarkeit in Endpunktaktivität und ermöglicht schnelle Bedrohungsreaktion.

- Setzen Sie EDR-Agenten auf allen Servern und Arbeitsplätzen ein, die das Legacy-System hosten oder darauf zugreifen. Stellen Sie sicher, dass Agenten mit den genutzten Legacy-Betriebssystemen kompatibel sind — manche Legacy-Systeme laufen auf älteren OS-Versionen, die spezifische EDR-Agent-Versionen erfordern.
- Konfigurieren Sie Baseline-Verhaltensprofile für jeden Endpunkt, um normale Aktivität von anomalem Verhalten zu unterscheiden. Legacy-Systeme haben oft vorhersehbare, repetitive Nutzungsmuster, die Anomalieerkennung besonders effektiv machen.
- Aktivieren Sie Prozessüberwachung, um unautorisierte Prozessausführung zu erkennen, einschließlich Skripten, unautorisierten Binärdateien und Prozessen, die nicht auf Produktionsservern laufen sollten.
- Implementieren Sie Dateiintegritätsüberwachung für kritische Systemdateien, Anwendungsbinärdateien und Konfigurationsdateien. Legacy-Systeme sind attraktive Ziele für persistente Bedrohungen, die Systemdateien modifizieren, um Zugriff aufrechtzuerhalten.
- Konfigurieren Sie automatisierte Reaktionsaktionen für Bedrohungen hoher Konfidenz: Isolieren Sie den Endpunkt vom Netzwerk, terminieren Sie bösartige Prozesse und alarmieren Sie das Sicherheitsteam. Erstellen Sie für Erkennungen niedrigerer Konfidenz Alarme ohne automatisierte Reaktion, um legitime Legacy-System-Operationen nicht zu stören.
- Integrieren Sie EDR-Telemetrie mit zentralisiertem SIEM zur Korrelation über Endpunkte und Netzwerkaktivität hinweg, was die Erkennung lateraler Bewegung und mehrstufiger Angriffe ermöglicht.

## Tradeoffs ⇄

> EDR bietet Echtzeit-Bedrohungssichtbarkeit und schnelle Reaktionsfähigkeit für Legacy-System-Endpunkte, erfordert aber kompatible Agenten und sorgfältige Abstimmung, um Falsch-Positive zu vermeiden.

**Vorteile:**

- Erkennt Kompromittierungen in Echtzeit statt nachträglich, was sofortige Eindämmung ermöglicht und die Auswirkung von Sicherheitsvorfällen reduziert.
- Bietet detaillierte forensische Daten (Prozessbäume, Dateimodifikationen, Netzwerkverbindungen), die die Vorfalluntersuchung unterstützen.
- Ermöglicht automatisierte Eindämmungsaktionen, die Schaden begrenzen, selbst wenn Sicherheitspersonal nicht sofort verfügbar ist.
- Kompensiert das Fehlen von Sicherheitskontrollen auf Anwendungsebene in Legacy-Systemen, indem Endpunktverhalten überwacht wird.

**Kosten und Risiken:**

- EDR-Agenten verbrauchen CPU- und Speicherressourcen auf Legacy-Servern, die möglicherweise bereits ressourcenbeschränkt sind.
- Kompatibilitätsprobleme mit Legacy-Betriebssystemen oder -Anwendungen können die EDR-Funktionalität einschränken oder Systeminstabilität verursachen.
- Hohe Falsch-Positiv-Raten auf Legacy-Systemen mit ungewöhnlichen, aber legitimen Verhaltensmustern können das Sicherheitsteam überwältigen.
- Automatisierte Reaktionsaktionen (Endpunktisolation) können ungeplante Ausfälle verursachen, wenn sie fälschlich auf kritischen Legacy-System-Servern ausgelöst werden.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie EDR Legacy-System-Umgebungen schützt.

Ein Legacy-Buchhaltungssystem läuft auf einer Windows-Server-2012-Instanz, die wegen Bedenken bezüglich Anwendungskompatibilität seit über einem Jahr keine Sicherheitspatches erhalten hat. Ein Angreifer nutzt eine bekannte Remote-Code-Execution-Schwachstelle aus und etabliert eine persistente Hintertür. Ohne EDR bleibt die Kompromittierung drei Monate unentdeckt, während derer der Angreifer Finanzdaten exfiltriert. Nach dem Einsatz von EDR erkennt der Agent innerhalb von Minuten einen ähnlichen Kompromittierungsversuch auf einem anderen Legacy-Server — die EDR identifiziert einen ungewöhnlichen PowerShell-Prozess, gestartet vom Service-Konto der Buchhaltungsanwendung, ein Verhalten, das im Baseline-Profil des Endpunkts nie aufgetreten ist. Der Agent isoliert den Endpunkt automatisch und alarmiert das Sicherheitsteam, das den Vorfall innerhalb von 30 Minuten eindämmt.

Ein Legacy-Fertigungssteuerungssystem läuft auf dedizierten Arbeitsplätzen, verbunden sowohl mit dem Unternehmensnetzwerk als auch dem Werkshallennetzwerk. Der EDR-Einsatz auf diesen Arbeitsplätzen enthüllt, dass ein Arbeitsplatz in regelmäßigen Abständen mit einer unbekannten externen IP-Adresse kommuniziert — ein Muster, das mit Command-and-Control-Beaconing übereinstimmt. Die Untersuchung enthüllt, dass ein für Datenübertragung zwischen Systemen genutztes USB-Laufwerk mit Malware infiziert war, die Persistenz auf dem Arbeitsplatz etablierte. Die EDR-Eindämmung blockiert die externe Kommunikation sofort, und die forensische Telemetrie des EDR-Agenten liefert die vollständige Infektionszeitlinie, was dem Team erlaubt zu verifizieren, dass keine Werkshallensysteme kompromittiert wurden.
