---
title: Circuit Breaker
description: Mechanismus zum Schutz vor Fehlern und Überlast in verteilten Systemen.
category:
- Architecture
problems:
- cascade-failures
- service-timeouts
- external-service-delays
- system-outages
- thread-pool-exhaustion
- upstream-timeouts
- single-points-of-failure
- service-discovery-failures
layout: solution
lang: de
en_slug: circuit-breaker
related_solutions:
- slug: retry
  similarity: 0.8
- slug: failover-mechanisms
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: error-handling
  similarity: 0.75
- slug: isolation-of-faulty-components
  similarity: 0.7
- slug: resilience
  similarity: 0.7
---

## Description

Ein Circuit Breaker ist ein Schutzwrapper, platziert um einen Aufruf an einen externen Service oder eine Abhängigkeit, der wiederholte Fehler überwacht und, sobald eine Fehlerschwelle überschritten wird, „öffnet", um weitere Aufrufe vollständig zu stoppen, schnell mit einer Fallback-Antwort zu scheitern, statt weiterhin einen Service zu treffen, von dem bekannt ist, dass er ungesund ist. Nach einem konfigurierten Intervall wechselt er in einen „halb-offenen" Zustand, der eine kleine Anzahl von Testanfragen durchlässt, um zu prüfen, ob sich die Abhängigkeit erholt hat, und schließt sich wieder, wenn sie erfolgreich sind. Dies zielt direkt auf ein übliches Legacy-Systemfehlermuster ab, bei dem sich synchrone Aufrufe an einen kämpfenden nachgelagerten Service in einem Thread-Pool oder Verbindungspool anhäufen, während jeder Aufrufer blockiert und auf ein Timeout wartet, was schließlich diese Ressource erschöpft und einen Ausfall in einer Komponente verursacht, die selbst nichts falsch macht. Weil viele Legacy-Systeme mit eng gekoppelten, synchronen Integrationspunkten ohne Isolation zwischen ihnen gebaut wurden, kann eine einzelne langsame oder ausfallende Abhängigkeit sonst zu einem systemweiten Ausfall kaskadieren, weit größer als das ursprüngliche Problem. Indem er schnell scheitert und ein definiertes Fallback ersetzt — gecachte Daten, eine degradierte Antwort oder einen klaren Fehler — verwandelt der Circuit Breaker einen unbegrenzten, ressourcenverbrauchenden Fehler in einen begrenzten, vorhersehbaren. Seine Effektivität hängt davon ab, sinnvolles Fallback-Verhalten für jeden geschützten Aufruf zu designen und Schwellen gegen das tatsächliche Verhalten der Abhängigkeit abzustimmen, da ein schlecht konfigurierter Breaker entweder nicht rechtzeitig auslösen oder legitimen Traffic während gewöhnlicher vorübergehender Störungen ablehnen kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie alle externen Serviceaufrufe und Inter-Service-Kommunikationspunkte, die blockieren oder fehlschlagen könnten
- Umschließen Sie kritische externe Aufrufe mit einer Circuit-Breaker-Bibliothek (z. B. Resilience4j, Polly, Hystrix)
- Konfigurieren Sie Fehlerschwellen, die den Circuit zum Öffnen bringen und weitere Aufrufe an den fehlerhaften Service verhindern
- Definieren Sie Fallback-Verhalten für jeden Circuit Breaker: gecachte Daten, degradierte Funktionalität oder eine nutzerfreundliche Fehlermeldung
- Setzen Sie angemessene Timeout-Fenster für halb-offene Zustände, die periodisches Testen des wiederhergestellten Services erlauben
- Fügen Sie Monitoring-Dashboards hinzu, die Circuit-Breaker-Zustände und Auslösungszählungen für operative Sichtbarkeit zeigen
- Stimmen Sie Circuit-Breaker-Parameter basierend auf beobachtetem Serviceverhalten und SLA-Anforderungen ab

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verhindert kaskadierende Ausfälle, indem Aufrufe an fehlerhafte nachgelagerte Services gestoppt werden
- Erlaubt dem System, elegant zu degradieren statt vollständig auszufallen
- Gibt fehlerhaften Services Zeit zur Erholung, ohne von Wiederholungsstürmen überwältigt zu werden
- Verbessert die Systemreaktionsfähigkeit, indem schnell gescheitert wird statt auf Timeouts zu warten

**Kosten und Risiken:**
- Fallback-Verhalten muss sorgfältig designt werden, um Dateninkonsistenzen zu vermeiden
- Offene Circuits könnten legitime Anfragen während vorübergehender Fehler ablehnen
- Fügt der Codebasis Komplexität hinzu und erfordert sorgfältige Konfigurationsabstimmung
- Circuit-Breaker-Zustand kann zugrunde liegende Probleme verdecken, wenn Monitoring unzureichend ist
- Halb-offene Prüflogik muss getestet werden, um korrekte Wiederherstellungserkennung sicherzustellen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Auftragsverarbeitungssystem machte synchrone Aufrufe an einen Bestandsservice, ein Zahlungsgateway und einen Versandanbieter. Als der Versandanbieter einen Ausfall erlebte, füllte sich der Thread-Pool des Auftragsservices mit blockierten Anfragen, die auf das Timeout der Versand-API warteten, was schließlich den gesamten Auftragsfluss unresponsiv machte. Das Team fügte Resilience4j-Circuit-Breaker um jeden externen Aufruf hinzu. Als sich der Versand-Circuit nach fünf aufeinanderfolgenden Fehlern öffnete, wurden Bestellungen akzeptiert, mit Versand geplant für spätere Verarbeitung statt den gesamten Checkout zu blockieren. Der halb-offene Zustand des Circuit Breakers erkannte automatisch, als sich der Versandanbieter erholte, und nahm den normalen Betrieb ohne manuellen Eingriff wieder auf.
