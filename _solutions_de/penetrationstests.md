---
title: Penetrationstests
description: Aufdeckung von Sicherheitslücken durch simulierte Angriffe.
category:
- Security
- Testing
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- data-protection-risk
- insufficient-testing
- quality-blind-spots
- session-management-issues
layout: solution
lang: de
en_slug: penetration-tests
related_solutions:
- slug: dynamic-code-analysis
  similarity: 0.8
- slug: security-tests-by-external-parties
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.8
- slug: fuzz-testing
  similarity: 0.8
- slug: honeypots
  similarity: 0.75
- slug: patch-management
  similarity: 0.75
---

## Description

Ein Penetrationstest simuliert einen echten Angreifer, der gegen das System arbeitet — Schwächen verkettet, Geschäftslogik testet und Privilege Escalation versucht —, um ausnutzbare Schwachstellen zu finden, die automatisierte Scanner und Code-Reviews allein nicht aufdecken können. Legacy-Systeme sind ein häufiges Ziel für genau die Art von Befunden, auf die diese Methode spezialisiert ist: Standardanmeldedaten, die auf vergessenen administrativen Schnittstellen zurückgeblieben sind, veraltete Protokollkonfigurationen und Geschäftslogikfehler in Arbeitsabläufen, die niemand seit Jahren genau untersucht hat — nichts davon würde ein statischer Scan erfassen, weil es das Urteilsvermögen eines Angreifers erfordert, um es als ausnutzbar zu erkennen. Da die Übung punktuell und ressourcenintensiv ist, funktioniert sie am besten als wiederkehrende Praxis — mindestens jährlich und nach bedeutenden Änderungen —, wobei frühere Befunde nachverfolgt werden, um zu verifizieren, dass die Behebung den ausnutzbaren Pfad tatsächlich geschlossen und nicht nur verborgen hat.

## How to Apply ◆

> Legacy-Systeme enthalten oft Sicherheitslücken, die automatisierte Scanner übersehen, weil sie Verständnis der Geschäftslogik, Verkettung mehrerer Schwächen oder Ausnutzungstechniken erfordern, die für den Legacy-Technologie-Stack spezifisch sind. Penetrationstests simulieren reale Angriffe, um ausnutzbare Schwachstellen zu entdecken.

- Definieren Sie den Umfang des Penetrationstests basierend auf dem Risikoprofil des Legacy-Systems: welche Komponenten, Schnittstellen und Netzwerksegmente im Umfang sind, welche Testmethoden erlaubt sind und was einen Befund gegenüber erwartetem Verhalten ausmacht.
- Führen Sie sowohl authentifizierte als auch unauthentifizierte Tests durch. Authentifizierte Tests (mit gültigen Anmeldedaten auf verschiedenen Privilegienebenen) offenbaren Autorisierungsumgehung und Privilege-Escalation-Probleme, die von außerhalb der Anwendung unsichtbar sind.
- Konzentrieren Sie Tests auf legacy-spezifische Risikobereiche: Standardanmeldedaten auf administrativen Schnittstellen, veraltete TLS/SSL-Konfigurationen, Injection-Schwachstellen in Legacy-Code, der vor modernen Frameworks entstand, und unsichere Deserialisierung in Legacy-APIs.
- Testen Sie Geschäftslogik-Schwachstellen, die automatisierte Scanner nicht erkennen können: Preismanipulation, Umgehung von Arbeitsabläufen, Race Conditions in mehrstufigen Transaktionen und Zugriffskontrolllücken zwischen verschiedenen Nutzerrollen.
- Führen Sie Tests auf Netzwerkebene durch, um unnötige offene Ports, falsch konfigurierte Dienste und Netzwerkpfade zu identifizieren, die nicht existieren sollten. Legacy-Infrastruktur hat oft Dienste angesammelt, die für Debugging oder Tests gestartet und nie entfernt wurden.
- Klassifizieren Sie Befunde nach Schweregrad (kritisch, hoch, mittel, niedrig) und Ausnutzbarkeit (leicht ausnutzbar vs. spezifische Bedingungen erfordernd). Priorisieren Sie Behebung basierend auf der Kombination aus Auswirkung und Ausnutzbarkeit.
- Planen Sie regelmäßige Penetrationstests (mindestens jährlich und nach bedeutenden Änderungen) und verfolgen Sie die Behebung von Befunden aus früheren Tests. Verifizieren Sie, dass Fixes die Schwachstelle tatsächlich adressieren, statt sie nur zu verschleiern.

## Tradeoffs ⇄

> Penetrationstests bieten eine realistische Bewertung ausnutzbarer Schwachstellen aus der Perspektive eines Angreifers, sind aber punktuell und ressourcenintensiv.

**Vorteile:**

- Entdeckt ausnutzbare Schwachstellen, die automatisierte Scanner und Code-Reviews übersehen, insbesondere Geschäftslogikfehler und mehrstufige Angriffsketten.
- Bietet die Perspektive eines Angreifers auf die Sicherheitslage des Systems und offenbart, welche Schwachstellen praktisch ausnutzbar sind gegenüber nur theoretischen.
- Validiert, dass Sicherheitskontrollen (Firewalls, WAFs, Authentifizierung, Autorisierung) unter feindlichen Bedingungen tatsächlich wie beabsichtigt funktionieren.
- Produziert priorisierte Behebungsanleitung basierend auf echter Ausnutzbarkeit statt theoretischer Schweregradbewertungen.

**Kosten und Risiken:**

- Penetrationstests sind teuer und ressourcenintensiv und erfordern qualifizierte Sicherheitsfachleute mit Kenntnis des Legacy-Technologie-Stacks.
- Tests sind punktuell — neue Schwachstellen, die nach dem Test eingeführt werden, werden bis zum nächsten Einsatz nicht erkannt.
- Aggressive Testtechniken können Dienstunterbrechungen, Datenkorruption oder Denial-of-Service in Legacy-Systemen verursachen, die nicht darauf ausgelegt sind, feindliche Eingaben elegant zu handhaben.
- Befunde aus Penetrationstests können für Teams mit begrenzter Kapazität zur Sicherheitsbehebung überwältigend sein, was sorgfältige Priorisierung erfordert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Penetrationstests kritische Schwachstellen in Legacy-Systemen aufdecken.

Eine Legacy-Gesundheitsanwendung durchläuft nach 10 Jahren in Produktion ihren ersten Penetrationstest. Die Tester entdecken, dass das Session-Management der Anwendung vorhersagbare Session-IDs verwendet (sequenzielle Ganzzahlen), was einem Angreifer erlaubt, jede aktive Sitzung zu kapern, indem er die Session-ID erhöht. Sie finden außerdem, dass die „Passwort vergessen"-Funktion das tatsächliche Passwort im Klartext per E-Mail sendet (statt eines Reset-Links), was offenbart, dass Passwörter in umkehrbarer Verschlüsselung statt Einweg-Hashing gespeichert werden. Zusätzlich ist die administrative Schnittstelle vom Internet aus zugänglich ohne jegliche zusätzliche Authentifizierung über den gleichen Login hinaus, den normale Nutzer verwenden. Jeder Befund wird als kritisch eingestuft, und der Behebungsplan priorisiert sie in dieser Reihenfolge: Beschränkung des administrativen Zugangs auf das interne Netzwerk (sofort), Implementierung kryptografischer Session-IDs (1 Woche) und Migration der Passwortspeicherung zu bcrypt mit einem Reset-Ablauf (1 Monat).

Eine Legacy-Handelsplattform im Finanzbereich erhält jährliche Penetrationstests. Der jüngste Test offenbart eine Race Condition in der Order-Placement-API: Durch gleichzeitiges Einreichen zweier Orders für dasselbe mengenbeschränkte Wertpapier kann ein Angreifer mehr Einheiten kaufen, als verfügbar sind, weil die Bestandsprüfung und die Bestandsdekrementierung nicht atomar sind. Die Tester demonstrieren, dass dies mit einfachen gleichzeitigen HTTP-Anfragen ausgenutzt werden kann, ohne jegliche Spezialwerkzeuge. Der Befund führt zur Implementierung datenbankseitiger Sperren bei Bestandsprüfungen, eine Schwachstelle, die automatisierte Schwachstellenscanner nie erkannt hätten, weil sie Verständnis der Geschäftslogik und die Fähigkeit erfordert, gleichzeitige Anfragenbehandlung zu testen.
