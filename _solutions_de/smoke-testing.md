---
title: Smoke Testing
description: Durchführung einer Reihe grundlegender Tests zur Verifikation
  der Kernfunktionalität eines Systems.
category:
- Testing
- Operations
problems:
- frequent-hotfixes-and-rollbacks
- regression-bugs
- deployment-risk
- high-defect-rate-in-production
- insufficient-testing
- release-instability
- fear-of-change
- missing-end-to-end-tests
- legacy-code-without-tests
- increased-manual-testing-effort
layout: solution
lang: de
en_slug: smoke-testing
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.8
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: regression-testing
  similarity: 0.75
- slug: blue-green-canary-deployments
  similarity: 0.75
- slug: functional-tests
  similarity: 0.75
- slug: continuous-deployment
  similarity: 0.75
---

## Description

Ein Smoke Test ist eine kleine, schnelle Suite von End-to-End-Prüfungen — typischerweise Login, Kerntransaktionen und andere Operationen abdeckend, deren Fehlschlag das System unbenutzbar machen würde —, die unmittelbar nach jedem Deployment ausgeführt wird, um zu bestätigen, dass das System grundlegend betriebsbereit ist, bevor irgendeine tiefere Verifikation stattfindet. Der Name stammt aus dem Hardware-Testing, wo das Einschalten eines Geräts und die Prüfung auf Rauch das erste, günstigste mögliche Signal ist, dass etwas katastrophal falsch ist. In Legacy-Systemen, denen umfassende automatisierte Testsuiten fehlen, bietet Smoke Testing ein ungewöhnlich günstiges Verhältnis von Schutz zu Aufwand: Es erfordert kein Verständnis des vollständigen Systemverhaltens, sondern nur die Identifikation der Handvoll Pfade, deren Bruch sofort und schwer störend wäre. In die Deployment-Pipeline als Gate eingebunden, das die Beförderung blockiert oder bei Fehlschlag automatischen Rollback auslöst, verwandelt es das, was sonst eine stundenlange Verzögerung zwischen einem defekten Deployment und seiner Entdeckung wäre — oft über eine Nutzerbeschwerde — in ein innerhalb von Minuten produziertes Signal. Dies macht Smoke Testing zu einer praktischen ersten Investition für Teams, die sonst Legacy-Änderungen auf Vertrauen basierend ausliefern, obwohl seine Oberflächlichkeit bedeutet, dass es bewusst unfähig ist, subtile Regressionen oder Grenzfall-Defekte zu erfassen, die weiterhin die tieferen Testschichten erfordern, die es ergänzen, nicht ersetzen soll.

## How to Apply ◆

> Legacy-Systemen fehlen oft umfassende Testsuiten, was jedes Deployment zu einem Glücksspiel macht. Smoke Tests bieten ein leichtgewichtiges Sicherheitsnetz, das verifiziert, dass die Kernfunktionalität nach dem Deployment funktioniert, und katastrophale Fehler erfasst, bevor sie Nutzer erreichen.

- Identifizieren Sie die 10-20 kritischsten nutzerseitigen Operationen im Legacy-System — Login, Kerntransaktionen, Schlüssel-API-Endpunkte, kritische Batch-Jobs. Diese bilden die anfängliche Smoke-Test-Suite und sollten die Pfade abdecken, die, wenn defekt, das System unbenutzbar machen würden.
- Implementieren Sie Smoke Tests als einfache, schnelle End-to-End-Prüfungen, die verifizieren, dass jede kritische Operation erfolgreich abgeschlossen wird. Smoke Tests sollten nicht erschöpfend sein; sie sollten in unter 5 Minuten laufen und bestätigen, dass das System grundlegend betriebsbereit ist.
- Integrieren Sie Smoke Tests in die Deployment-Pipeline, sodass sie automatisch nach jedem Deployment in jede Umgebung laufen. Ein fehlgeschlagener Smoke Test sollte die Beförderung zur nächsten Umgebung blockieren oder einen automatischen Rollback auslösen.
- Gestalten Sie Smoke Tests umgebungsagnostisch, indem Sie Endpunkte, Zugangsdaten und Testdaten parametrisieren. Dies erlaubt es, dieselbe Suite gegen Staging-, Pre-Produktions- und Produktionsumgebungen laufen zu lassen.
- Beziehen Sie Health Checks für kritische Abhängigkeiten (Datenbankverbindung, Verfügbarkeit externer Dienste, Message-Broker-Verbindung) als Teil der Smoke-Suite ein. Legacy-Systeme versagen oft still, wenn eine Abhängigkeit nicht verfügbar ist.
- Fügen Sie Smoke Tests für Legacy-Batch-Prozesse hinzu, indem Sie verifizieren, dass ein minimaler Batch-Lauf ohne Fehler abschließt. Batch-Fehlschläge in Legacy-Systemen bleiben oft unentdeckt, bis nachgelagerte Konsumenten Stunden später fehlende Daten bemerken.
- Planen Sie periodische Smoke-Test-Ausführung in Produktion (alle 15-30 Minuten), um Umgebungsprobleme, Zertifikatsabläufe oder Ressourcenerschöpfung zu erkennen, bevor Nutzer betroffen sind.

## Tradeoffs ⇄

> Smoke Tests bieten schnelle, hochvertrauenswürdige Verifikation der Kernfunktionalität mit minimaler Investition, sind aber bewusst oberflächlich und können umfassendes Testing nicht ersetzen.

**Vorteile:**

- Erfassen katastrophale Deployment-Fehlschläge innerhalb von Minuten statt Stunden, was die mittlere Erkennungszeit für defekte Deployments drastisch reduziert.
- Bieten einen kostengünstigen Ausgangspunkt für das Testen von Legacy-Systemen ohne bestehende Testsuite, was mit minimalem Aufwand sofortigen Wert liefert.
- Bauen Deployment-Vertrauen schrittweise auf und helfen Teams, von angstgetriebener manueller Verifikation zu automatisierten Sicherheitsprüfungen zu wechseln.
- Ermöglichen schnellere Rollback-Entscheidungen durch klare Bestanden/Fehlgeschlagen-Signale unmittelbar nach dem Deployment.

**Kosten und Risiken:**

- Smoke Tests verifizieren nur, dass das System startet und grundlegende Operationen handhabt; sie übersehen subtile Fehler, Performance-Regressionen und Grenzfälle, die tieferes Testing erfordern.
- Falsches Vertrauen kann entstehen, wenn Teams bestandene Smoke Tests als Beweis behandeln, dass ein Deployment sicher ist, und dabei die Notwendigkeit breiterer Testabdeckung vernachlässigen.
- Smoke Tests in Produktion erfordern sorgfältiges Design, um Nebenwirkungen zu vermeiden — das Erstellen von Testdaten, das Auslösen von Benachrichtigungen oder die Beeinträchtigung echter Nutzer.
- Die Pflege von Smoke Tests für sich schnell entwickelnde Systeme erfordert laufenden Aufwand, um sie mit der aktuellen Funktionalität abgestimmt zu halten.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Smoke Tests kritische Fehler in Legacy-Systemen erfassen.

Eine Regierungsbehörde setzt monatlich Updates für ein Legacy-Bürgerdienstportal ein. Jedes Deployment beinhaltet die Aktualisierung von Konfigurationsdateien, Datenbankschemata und Anwendungsbinärdateien über mehrere Server hinweg. Bei drei Gelegenheiten im vergangenen Jahr brachen Deployments den Login-Flow aufgrund fehlkonfigurierter Authentifizierungseinstellungen, aber das Problem wurde erst entdeckt, als Bürger am nächsten Morgen beim Helpdesk anriefen. Das Team implementiert eine Smoke-Test-Suite, die unmittelbar nach dem Deployment versucht, sich mit einem Testkonto anzumelden, ein Beispiel-Antragsformular einzureichen und zu verifizieren, dass das Hauptdashboard korrekt lädt. Während des nächsten Deployments erkennt der Smoke Test, dass der Login-Endpunkt aufgrund einer fehlenden Umgebungsvariable einen 500-Fehler zurückgibt. Das Deployment wird innerhalb von 10 Minuten zurückgerollt, bevor irgendein Bürger betroffen ist. Die gesamte Smoke-Suite läuft in 90 Sekunden.

Ein Fertigungsunternehmen betreibt ein Legacy-ERP-System, bei dem nächtliche Batch-Jobs Produktionsaufträge, Bestandsaktualisierungen und Lieferantenrechnungen verarbeiten. Fehlschläge in diesen Batch-Jobs werden typischerweise erst am nächsten Nachmittag entdeckt, wenn Lagermitarbeiter fehlende Kommissionierlisten bemerken. Das Team erstellt einen Smoke Test, der nach jedem Batch-Zyklus läuft und verifiziert, dass mindestens ein Auftrag verarbeitet wurde, Bestandszahlen aktualisiert wurden und keine Fehlerdatensätze in die Batch-Log-Tabelle geschrieben wurden. Als eine Fehlkonfiguration des Datenbankverbindungspools dazu führt, dass der Bestands-Batch still fehlschlägt, erkennt der Smoke Test die Null-Update-Bedingung innerhalb von 15 Minuten und alarmiert den Bereitschaftsingenieur, der das Problem behebt, bevor die Frühschicht beginnt.
