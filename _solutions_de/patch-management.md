---
title: Patch-Management
description: Zeitnahe Anwendung von Sicherheitsupdates und Patches.
category:
- Security
- Operations
problems:
- obsolete-technologies
- system-outages
- data-protection-risk
- regulatory-compliance-drift
- deployment-risk
- dependency-version-conflicts
- technology-lock-in
- fear-of-change
layout: solution
lang: de
en_slug: patch-management
related_solutions:
- slug: regular-maintenance-and-updates
  similarity: 0.8
- slug: incident-response-measures
  similarity: 0.8
- slug: vulnerability-scans
  similarity: 0.8
- slug: security-hardening-process
  similarity: 0.75
- slug: security-audits
  similarity: 0.75
- slug: honeypots
  similarity: 0.75
---

## Description

Patch-Management ist der systematische Prozess des Nachverfolgens, Bewertens, Testens und Anwendens von Sicherheitsupdates über jede Softwarekomponente eines Systems, statt Updates ad hoc oder gar nicht geschehen zu lassen. Legacy-Systeme sind hier überproportional exponiert: dieselbe Angst vor dem Brechen undokumentierter, ungetesteter Funktionalität, die Teams vom Refactoring abhält, hält sie auch vom Patchen ab, was bekannte, öffentlich offengelegte Schwachstellen jahrelang unadressiert lässt. Ein disziplinierter Prozess — Inventarisierung von Komponenten gegen Schwachstellendatenbanken, Testen von Patches in einer Staging-Umgebung vor der Produktion und Definition von Rollback-Prozeduren — schließt dieses Expositionsfenster bewusst statt durch Glück. Für Komponenten, die vollständig das Lebensende erreicht haben und überhaupt nicht mehr gepatcht werden können, sollte derselbe Prozess kompensierende Kontrollen wie Netzwerkisolation oder virtuelles Patchen auslösen, da Patchen allein Software nicht schützen kann, deren Hersteller den Support eingestellt hat.

## How to Apply ◆

> Legacy-Systeme bleiben häufig ungepatcht aufgrund von Angst vor Funktionsbruch, fehlender Testinfrastruktur oder Herstelleraufgabe. Patch-Management etabliert systematische Prozesse zur Bewertung, zum Testen und zur Anwendung von Sicherheitsupdates.

- Inventarisieren Sie alle Softwarekomponenten im Legacy-System (Betriebssysteme, Anwendungs-Frameworks, Bibliotheken, Datenbanken, Middleware) und verfolgen Sie deren aktuelle Versionen gegen bekannte Schwachstellendatenbanken (CVE, NVD).
- Etablieren Sie ein Patch-Klassifizierungs- und Priorisierungsschema: kritische Schwachstellen in internetseitigen Komponenten brauchen sofortige Aufmerksamkeit, während Probleme geringerer Schwere in isolierten internen Komponenten einem regulären Zeitplan folgen können.
- Erstellen Sie eine Testpipeline für Patches: wenden Sie Patches auf eine Staging-Umgebung an, die die Produktion spiegelt, führen Sie Regressionstests durch und verifizieren Sie, dass die Legacy-Anwendung korrekt funktioniert, bevor Sie in Produktion anwenden.
- Definieren Sie Patch-Fenster und Rollback-Prozeduren. Für Legacy-Systeme, die für das Patchen Ausfallzeit benötigen, planen Sie regelmäßige Wartungsfenster. Dokumentieren Sie für jeden Patch die Rollback-Prozedur, falls das Update Probleme verursacht.
- Implementieren Sie für Legacy-Systeme, die auf Betriebssystemen oder Frameworks am Lebensende laufen, die keine Patches mehr erhalten, kompensierende Kontrollen: Netzwerkisolation, virtuelles Patchen durch WAF-Regeln, Anwendungs-Whitelisting und verstärkte Überwachung.
- Automatisieren Sie Schwachstellenscans, um kontinuierlich ungepatchte Komponenten und neu offengelegte Schwachstellen zu identifizieren. Planen Sie wöchentliche oder tägliche Scans und integrieren Sie Befunde in die Arbeitswarteschlange des Teams.
- Verfolgen Sie Patch-Compliance-Kennzahlen (Zeit von Offenlegung bis Patch-Anwendung, Prozentsatz aktueller Systeme) und berichten Sie diese an das Management, um organisatorisches Engagement für das Patchen zu erhalten.

## Tradeoffs ⇄

> Zeitnahes Patch-Management schließt bekannte Schwachstellenfenster und reduziert die Angriffsfläche, erfordert aber Testinfrastruktur, Wartungsfenster und sorgfältiges Risikomanagement.

**Vorteile:**

- Schließt bekannte Schwachstellenfenster, bevor Angreifer sie ausnutzen können, was die Angriffsfläche des Systems direkt reduziert.
- Erhält die Herstellersupport-Berechtigung, indem Software innerhalb unterstützter Versionsbereiche gehalten wird.
- Unterstützt die Compliance mit Sicherheitsstandards, die zeitnahe Anwendung von Sicherheitspatches vorschreiben.
- Reduziert die Anhäufung technischer Schulden durch Versionsdrift, indem ein regelmäßiger Update-Rhythmus beibehalten wird.

**Kosten und Risiken:**

- Patches können Legacy-Anwendungsfunktionalität brechen, besonders wenn die Anwendung von spezifischem Verhalten der zugrunde liegenden Plattform abhängt, das der Patch ändert.
- Legacy-Systemen fehlt oft die automatisierte Testinfrastruktur, die nötig ist, um zuverlässig zu verifizieren, dass Patches keine Regressionen einführen.
- Patchen kann Ausfallzeit erfordern, was für Systeme mit hohen Verfügbarkeitsanforderungen schwer zu planen ist.
- Software am Lebensende, die keine Patches mehr erhält, kann nicht durch Patchen behoben werden, was Migration oder kompensierende Kontrollen erfordert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Patch-Management Risiko in Legacy-Systemen reduziert.

Ein Legacy-Webserver, der Apache 2.2 betreibt, wurde seit 3 Jahren nicht aktualisiert. Eine kritische Schwachstelle für Remote Code Execution (CVE) wird offengelegt, die diese Version betrifft. Ohne einen Patch-Management-Prozess bleibt die Schwachstelle unbemerkt, bis ein Penetrationstest sie 6 Monate später entdeckt. Nach der Implementierung von Patch-Management abonniert das Team Schwachstellen-Feeds für alle Softwarekomponenten in seinem Stack. Als eine neue kritische Apache-Schwachstelle offengelegt wird, markiert automatisiertes Scanning ihre Server innerhalb von 24 Stunden. Das Team wendet den Patch auf die Staging-Umgebung an, führt automatisierte Regressionstests durch, verifiziert, dass die Legacy-Anwendung korrekt funktioniert, und stellt innerhalb von 72 Stunden nach Offenlegung in Produktion bereit — deutlich bevor Exploit-Code breit verfügbar wird. Das Team erstellt außerdem einen Plan, von Apache 2.2 auf eine derzeit unterstützte Version zu migrieren, um fortlaufende Patch-Verfügbarkeit sicherzustellen.

Eine Legacy-Unternehmensanwendung hängt von einer Java-7-Laufzeitumgebung ab, die vor Jahren das Lebensende erreichte. Keine Sicherheitspatches sind verfügbar, und ein Upgrade auf Java 8+ erfordert aufgrund veralteter API-Nutzung erhebliche Anwendungsänderungen. Das Team implementiert kompensierende Kontrollen, während es das Java-Upgrade plant: Es stellt eine virtuelle Patching-Schicht mittels der WAF bereit, die die Ausnutzung bekannter Java-7-Schwachstellen blockiert, beschränkt den Netzwerkzugang des Anwendungsservers auf nur benötigte Ziele und implementiert Anwendungs-Whitelisting, um unautorisierte Codeausführung zu verhindern. Diese kompensierenden Kontrollen bieten Schutz während des 8-monatigen Java-Migrationsprojekts, und Schwachstellenscans bestätigen, dass die bekannten Java-7-CVEs über den Netzwerkpfad nicht ausgenutzt werden können, obwohl die Laufzeitumgebung ungepatcht bleibt.
