---
title: Erhöhte Fehlerraten
description: Ein ungewöhnlicher oder anhaltender Anstieg der Häufigkeit von Fehlern,
  die von einer Anwendung oder einem Dienst gemeldet werden.
category:
- Code
related_problems:
- slug: increased-bug-count
  similarity: 0.7
- slug: high-bug-introduction-rate
  similarity: 0.7
- slug: system-outages
  similarity: 0.65
- slug: service-timeouts
  similarity: 0.65
- slug: inadequate-error-handling
  similarity: 0.65
- slug: external-service-delays
  similarity: 0.6
solutions:
- observability-and-monitoring
- confirmation-dialogs
- dead-letter-queue
- feedback
- form-design
- input-constraints-and-defaults
- plausibility-checks
- retry
- root-cause-analysis
- error-handling
- error-logs
- error-reporting-and-analysis
- real-time-input-validation
- understandable-error-messages
- undo-and-redo
- visual-hierarchy
layout: problem
lang: de
en_slug: increased-error-rates
---

## Description
Eine erhöhte Fehlerrate ist ein klares Zeichen dafür, dass mit einer Anwendung etwas nicht stimmt. Dies kann durch verschiedene Faktoren verursacht werden, von einem kürzlichen Deployment, das einen Fehler einführte, bis zu einem Problem mit einem nachgelagerten Dienst. Ein plötzlicher Anstieg der Fehlerrate sollte als ernstes Problem behandelt werden, da er erhebliche Auswirkungen auf das Nutzererlebnis und die Stabilität des Systems haben kann. Ein robustes Monitoring- und Alerting-System ist essenziell, um erhöhte Fehlerraten zeitnah zu erkennen und darauf zu reagieren.

## Indicators ⟡
- Es zeigt sich eine hohe Anzahl an Fehlern in den Logs.
- Das Monitoring-System löst Alerts für überschrittene Fehlerschwellenwerte aus.
- Es kommen Beschwerden von Nutzern über Fehler.
- Die Anwendung ist langsam oder nicht verfügbar.

## Symptoms ▲

- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Häufige Fehler verschlechtern das Nutzererlebnis, was zu Frustration und Vertrauensverlust in das System führt.
- [Erhöhte Last im Kundensupport](erhoehte-last-im-kundensupport.md)
<br/>  Nutzer, die auf Fehler stoßen, kontaktieren den Support, was das Ticket-Volumen in die Höhe treibt.
- [Negatives Nutzerfeedback](negatives-nutzerfeedback.md)
<br/>  Nutzer, die häufige Fehler erleben, geben negatives Feedback zur Systemzuverlässigkeit und -qualität.

## Causes ▼

- [ABI-Kompatibilitätsprobleme](abi-kompatibilitaetsprobleme.md)
<br/>  Laufzeitfehler durch ABI-Diskrepanzen führen zu erhöhten Fehlerraten, während Funktionsaufrufe unerwartete Werte zurückgeben oder abstürzen.
- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Schlechte Fehlerbehandlung erlaubt es Ausfällen, sich fortzupflanzen, statt elegant abgefangen und gehandhabt zu werden.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Eine hohe Rate neu eingeführter Fehler bei Änderungen trägt direkt zu mehr Laufzeitfehlern bei.
- [Falsche maximale Connection-Pool-Größe](falsche-maximale-connection-pool-groesse.md)
<br/>  Falsch konfigurierte Connection Pools verursachen Verbindungserschöpfung oder -ablehnung, was Anwendungsfehler erzeugt.
- [Datenbankverbindungslecks](datenbankverbindungslecks.md)
<br/>  Ausgelaufene Verbindungen erschöpfen den Pool über die Zeit, was zu einer wachsenden Anzahl verbindungsbezogener Fehler führt.
- [Service-Timeouts](service-timeouts.md)
<br/>  Erhöhte Fehlerraten begleiten oft kaskadierende Ausfälle, die Service-Timeouts über abhängige Systeme hinweg verursachen.
- [Release-Instabilität](release-instabilitaet.md)
<br/>  Spitzen bei Fehlerraten nach Deployments deuten darauf hin, dass Releases instabil sind und Produktionsprobleme verursachen.

## Detection Methods ○

- **Application Performance Monitoring (APM):** APM-Werkzeuge verfolgen Fehlerraten und können oft die genaue Codezeile oder den Dienst lokalisieren, der den Fehler verursacht.
- **Log-Aggregation und -Analyse:** Zentralisierte Logging-Systeme (z. B. ELK-Stack, Splunk) erlauben einfaches Suchen, Filtern und Visualisieren von Fehlerlogs.
- **Metriken und Alerting:** Überwachung von Fehlerraten (z. B. HTTP-5xx-Fehler, Ausnahmezahlen) und Einrichtung von Alerts für Spitzen.
- **Synthetisches Monitoring:** Automatisierte Tests, die Nutzerinteraktionen simulieren, können Fehler erkennen, bevor echte Nutzer betroffen sind.
- **Nutzerfeedback-Kanäle:** Aktive Überwachung von Kundensupport-Tickets, sozialen Medien und anderen Feedback-Kanälen.

## Examples
Nach einem neuen Release beginnt ein E-Commerce-Checkout-Dienst, einen hohen Prozentsatz an 500-Fehlern zurückzugeben. Die Untersuchung zeigt eine Änderung in der Zahlungs-Gateway-API, die der neue Code nicht berücksichtigte, was zu ungültigen Anfragen führte. In einem anderen Fall sieht ein Microservice, der Bild-Uploads verarbeitet, plötzlich eine Fehlerspitze. Bei der Untersuchung stellt sich heraus, dass die Festplatte, auf der hochgeladene Bilder gespeichert werden, den Speicherplatz aufgebraucht hat, was Dateischreiboperationen fehlschlagen lässt. Erhöhte Fehlerraten sind oft das erste Symptom eines tieferliegenden zugrunde liegenden Problems. Schnelle Erkennung und Diagnose sind entscheidend, um die Auswirkung auf Nutzer und Geschäftsbetrieb zu minimieren.
