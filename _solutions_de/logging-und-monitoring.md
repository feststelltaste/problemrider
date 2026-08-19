---
title: Logging und Monitoring
description: Protokollierung und Überwachung sicherheitsrelevanter Ereignisse.
category:
- Security
- Operations
problems:
- monitoring-gaps
- insufficient-audit-logging
- slow-incident-resolution
- debugging-difficulties
- logging-configuration-issues
- log-spam
- excessive-logging
- log-injection-vulnerabilities
layout: solution
lang: de
en_slug: logging-and-monitoring
related_solutions:
- slug: security-monitoring
  similarity: 0.85
- slug: audit-trail-management
  similarity: 0.8
- slug: logging
  similarity: 0.8
- slug: authentication
  similarity: 0.8
- slug: honeypots
  similarity: 0.8
- slug: digital-forensics
  similarity: 0.75
---

## Description

Logging und Monitoring, im sicherheitstechnischen Sinn, ist die disziplinierte Erfassung sicherheitsrelevanter Ereignisse — Authentifizierungsversuche, Autorisierungsentscheidungen, Datenzugriff, administrative Aktionen, Konfigurationsänderungen — in einen strukturierten, zentralisierten, manipulationsresistenten Datensatz, der nahezu in Echtzeit durchsucht, korreliert und alarmiert werden kann. Der Mechanismus kombiniert eine definierte Logging-Richtlinie, die spezifiziert, welche Ereignisse erfasst werden müssen, strukturierte Felder (Identität, Quelle, Aktion, Ergebnis), die diese Ereignisse maschinell durchsuchbar machen, Weiterleitung an ein zentralisiertes SIEM, das Aktivität über Komponenten hinweg korreliert, und Erkennungsregeln, die rohe Ereignisse in handlungsleitende Alarme für Muster wie wiederholte fehlgeschlagene Anmeldungen oder ungewöhnliche Datenexportvolumina umwandeln. Legacy-Systeme haben dies typischerweise verkehrt herum: Sie protokollieren zu wenig von dem, was für Sicherheit zählt, weil Authentifizierungs- und Autorisierungsereignisse nie als aufzeichnungswert galten, als das System ursprünglich gebaut wurde, während sie gleichzeitig zu viel betriebliches Rauschen protokollieren, sodass selbst dort, wo ein echtes Sicherheitssignal existiert, es unter Routine-Debug-Ausgaben begraben und unmöglich zu durchsuchen ist. Diese Kombination bedeutet, dass Sicherheitsvorfälle in Legacy-Umgebungen häufig über längere Zeit unentdeckt bleiben — ein Credential-Stuffing-Angriff oder das noch aktive Konto eines ausgeschiedenen Mitarbeiters kann Wochen oder Monate bestehen, einfach weil nichts in den bestehenden Logs dafür ausgelegt war, dies aufzudecken. Diese Fähigkeit in ein Legacy-System einzubauen erfordert, sicherheitsrelevante Ereignisse bewusst von Betriebs-Logging zu trennen und Instrumentierung an allen Stellen nachzurüsten, an denen sensible Aktionen stattfinden, was invasiv ist, aber meist auch der einzige Weg ist, die forensische und Erkennungsfähigkeit zu gewinnen, die Compliance-Regime wie PCI DSS, HIPAA oder DSGVO verlangen.

## How to Apply ◆

> Legacy-Systeme protokollieren für Sicherheitszwecke oft zu wenig (fehlende Authentifizierungsereignisse, Zugriffsentscheidungen), während sie zu viel Rauschen protokollieren (ausführliche Debug-Ausgaben, redundante Health Checks). Sicherheitsfokussiertes Logging und Monitoring erfasst die richtigen Ereignisse und macht sie handlungsleitend.

- Definieren Sie eine Sicherheits-Logging-Richtlinie, die spezifiziert, welche Ereignisse protokolliert werden müssen: Authentifizierungsversuche (Erfolg und Fehlschlag), Autorisierungsentscheidungen, Datenzugriff und -änderungen, administrative Aktionen, Konfigurationsänderungen und sicherheitsrelevante Fehler.
- Implementieren Sie strukturiertes Logging mit konsistenten Feldern über alle Legacy-Systemkomponenten hinweg: Zeitstempel (UTC), Ereignistyp, Schweregrad, Nutzeridentität, Quell-IP, zugegriffene Ressource, durchgeführte Aktion und Ergebnis (Erfolg/Fehlschlag).
- Leiten Sie alle Sicherheits-Logs nahezu in Echtzeit an ein zentralisiertes Security Information and Event Management (SIEM) System weiter. Zentralisierte Aggregation ermöglicht Korrelation über Komponenten hinweg und verhindert Log-Manipulation auf kompromittierten Systemen.
- Erstellen Sie Erkennungsregeln und Alarme für sicherheitsrelevante Muster: mehrere fehlgeschlagene Anmeldeversuche, Zugriff auf sensible Ressourcen außerhalb der Geschäftszeiten, Privilegieneskalation, ungewöhnliche Datenexportvolumina und Konfigurationsänderungen durch unerwartete Nutzer.
- Implementieren Sie Log-Schutz, um Manipulation zu verhindern: Speichern Sie Logs in Append-Only-Speicher, hashen Sie Log-Einträge zur Integritätsprüfung, und beschränken Sie Schreibzugriff nur auf das Logging-Dienstkonto.
- Adressieren Sie Log-Rauschen, indem betriebliche Logs (Health Checks, Routine-Statusaktualisierungen) von Sicherheits-Logs (Authentifizierung, Autorisierung, Datenzugriff) gefiltert oder getrennt werden. Sicherheits-Logs müssen zuverlässig durchsuchbar sein, ohne im Betriebsrauschen begraben zu werden.
- Stellen Sie sicher, dass sensible Daten (Passwörter, Tokens, Kreditkartennummern, personenbezogene Daten) nie in Logs geschrieben werden. Implementieren Sie Log-Bereinigungsfilter, die sensible Felder maskieren oder redigieren, bevor sie die Logging-Pipeline erreichen.

## Tradeoffs ⇄

> Sicherheitslogging und -monitoring bieten Sichtbarkeit auf Bedrohungen und ermöglichen schnelle Vorfallreaktion, erfordern aber Investition in Infrastruktur, Feinabstimmung und geschulte Analysten.

**Vorteile:**

- Ermöglicht die Erkennung von Sicherheitsvorfällen, die präventive Kontrollen nicht blockieren können, was die Zeit zwischen Verstoß und Entdeckung verringert.
- Liefert forensische Beweise für Vorfalluntersuchungen, was Ursachenanalyse und juristische Verfahren unterstützt.
- Erfüllt Compliance-Anforderungen für sicherheitsbezogenes Ereignis-Logging und -Monitoring (PCI DSS, HIPAA, SOX, DSGVO).
- Schafft Rechenschaftspflicht, indem erfasst wird, wer was wann getan hat, was unautorisierte Aktionen abschreckt.

**Kosten und Risiken:**

- Sicherheitslogging erzeugt erhebliche Datenvolumina, die Speicherung, Aufbewahrungsmanagement und Verarbeitungsinfrastruktur erfordern.
- Ohne angemessene Feinabstimmung produzieren Monitoring-Systeme Alarmmüdigkeit durch Falsch-Positive, was Analysten dazu bringt, echte Bedrohungen zu übersehen.
- Das Nachrüsten von Sicherheitslogging in Legacy-Systeme erfordert Codeänderungen an vielen Stellen, mit dem Risiko, kritische Ereignistypen zu übersehen.
- Logs mit sensiblen Daten (falls die Bereinigung unvollständig ist) schaffen ein sekundäres Datenschutzrisiko.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Sicherheitslogging und -monitoring Bedrohungen in Legacy-Systemen erkennen.

Eine Legacy-Banking-Anwendung protokolliert nur erfolgreiche Transaktionen, aber keine fehlgeschlagenen Anmeldeversuche oder Autorisierungsablehnungen. Ein Angreifer führt einen Credential-Stuffing-Angriff gegen die Login-Seite durch und testet über zwei Tage 50.000 Nutzername-Passwort-Kombinationen. Weil fehlgeschlagene Anmeldungen nicht protokolliert werden, ist der Angriff für das Betriebsteam unsichtbar. Nach der Implementierung umfassenden Sicherheitsloggings konfiguriert das Team Alarme für Muster wie mehr als 10 fehlgeschlagene Anmeldeversuche von einer einzigen IP innerhalb von 5 Minuten und mehr als 5 fehlgeschlagene Versuche für einen einzigen Benutzernamen innerhalb einer Stunde. Der nächste Credential-Stuffing-Versuch löst innerhalb von 3 Minuten einen Alarm aus, und die angreifenden IP-Adressen werden automatisch an der WAF blockiert. Das Sicherheitsteam identifiziert 12 Konten, bei denen der Angreifer richtig geraten hatte, und erzwingt Passwort-Resets, bevor unautorisierte Transaktionen stattfinden.

Ein Legacy-Dokumentenmanagementsystem hat ausführliches Anwendungslogging, das 50 GB Logs pro Tag schreibt, was es unmöglich macht, nach sicherheitsrelevanten Ereignissen zu suchen. Fehlgeschlagene Zugriffsversuche, administrative Aktionen und Datenexporte sind mit Tausenden von Debug-Meldungen über Bild-Rendering und Dokumentformatierung vermischt. Das Team implementiert eine strukturierte Logging-Strategie, die Sicherheitsereignisse in einen dedizierten Log-Stream trennt, der an das SIEM weitergeleitet wird, während betriebliche Logs weiterhin in den bestehenden Log-Speicher fließen. Sicherheitsereignisse werden mit standardisierten Ereignistypen markiert, was dem SIEM erlaubt, Erkennungsregeln anzuwenden. Innerhalb der ersten Woche entdeckt das SIEM, dass das Konto eines ausgeschiedenen Mitarbeiters noch aktiv ist und auf vertrauliche Dokumente zugreift — ein Ereignis, das monatelang geschah, aber im täglichen Log-Volumen von 50 GB unsichtbar war.
