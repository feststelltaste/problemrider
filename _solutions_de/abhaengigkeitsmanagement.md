---
title: Abhängigkeitsmanagement
description: Systematisierung der Verwaltung und Aktualisierung externer Abhängigkeiten.
category:
- Dependencies
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/dependency-management/
problems:
- dependency-version-conflicts
- vendor-lock-in
- vendor-dependency-entrapment
- vendor-dependency
- technology-lock-in
- breaking-changes
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- shared-dependencies
- dependency-on-supplier
- technology-stack-fragmentation
- obsolete-technologies
- premature-technology-introduction
- vendor-relationship-strain
layout: solution
lang: de
en_slug: dependency-management-strategy
related_solutions:
- slug: continuous-dependency-updates
  similarity: 0.85
- slug: third-party-dependency-check
  similarity: 0.8
- slug: regular-maintenance-and-updates
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: secret-management
  similarity: 0.75
- slug: static-analysis-and-linting
  similarity: 0.75
---

## Description

Eine Abhängigkeitsmanagement-Strategie inventarisiert, aktualisiert und scannt systematisch die externen Bibliotheken eines Systems und ersetzt Jahre unverwalteter Drift — fixierte uralte Versionen, keine Lock-Dateien, keine Sichtbarkeit darüber, was ein transitiver Abhängigkeitsbaum überhaupt enthält — durch einen bewussten, richtliniengesteuerten Prozess. Legacy-Systeme sind, wo sich diese Drift am schwersten anhäuft, da Updates zu vermeiden, weil ein vergangener Versuch eine Integrationskrise verursachte, eine vollkommen rationale kurzfristige Reaktion ist, die sich zu genau dem mehrversionigen Upgrade-Projekt summiert, das alle zu vermeiden versuchten. Ein vollständiges Abhängigkeitsinventar zu erzeugen, Lock-Dateien einzuführen und eine Bibliothek nach der anderen gegen eine schriftliche Richtlinie zu aktualisieren verwandelt ein unbegrenztes, still anwachsendes Risiko in eine routinemäßige, sichtbare Wartungsaufgabe.

## How to Apply ◆

> Legacy-Systeme häufen häufig Jahre unverwalteter Abhängigkeits-Drift an — auf uralte Versionen fixierte Bibliotheken, keine Lock-Dateien und keine Sichtbarkeit transitiver Schwachstellen —, was systematisches Abhängigkeitsmanagement sowohl dringend als auch technisch herausfordernd zu einführen macht.

- Beginnen Sie damit, ein vollständiges Abhängigkeitsinventar einschließlich transitiver Abhängigkeiten mittels `mvn dependency:tree`, `npm ls` oder Äquivalent zu erzeugen; für viele Legacy-Systeme ist dies das erste Mal, dass der volle Umfang der Abhängigkeitsfläche sichtbar wird, und das Ergebnis ist oft alarmierend.
- Führen Sie Lock-Dateien (package-lock.json, Pipfile.lock, pom.xml mit fixierten Versionen) sofort ein und committen Sie sie in die Versionskontrolle — dies etabliert eine reproduzierbare Baseline, von der aus alle zukünftigen Änderungen bewusst statt zufällig vorgenommen werden können.
- Führen Sie einen Schwachstellen-Scanner (Dependabot, Snyk, OWASP Dependency-Check) gegen den aktuellen Abhängigkeitsbaum aus, bevor Sie Updates vornehmen; das Ziel ist, das aktuelle Risiko zu verstehen, nicht alles auf einmal zu beheben — triagieren Sie nach Schweregrad und beginnen Sie mit kritischen und hohen Befunden.
- Definieren Sie eine schriftliche Update-Richtlinie für das Team: kritische Sicherheitspatches innerhalb einer Woche, Minor-Version-Updates innerhalb eines Sprints, Major-Version-Upgrades vierteljährlich geplant — Legacy-Teams haben oft überhaupt keine Richtlinie und reagieren nur, wenn etwas bricht.
- Aktualisieren Sie Abhängigkeiten eine Bibliothek nach der anderen in kleinen Schritten, statt ein Massen-Upgrade von allem über Jahre Angehäuften zu versuchen; drei Major-Versionen eines Frameworks gleichzeitig zu überspringen ist ein Projekt, keine Aufgabe.
- Etablieren Sie Schwachstellen-Scanning in der CI-Pipeline als Qualitätstor, das Merges mit kritischen oder hochschweren ungepatchten Abhängigkeiten blockiert — dies verhindert, dass sich die Anhäufung nach der anfänglichen Bereinigung wiederholt.
- Prüfen Sie Lizenzen über den Abhängigkeitsbaum mittels Werkzeugen wie FOSSA oder license-checker; Legacy-Systeme enthalten oft GPL-lizenzierte Bibliotheken in kommerziellen Produkten wegen Entscheidungen, die Jahre zuvor ohne Rechtsprüfung getroffen wurden.
- Überwachen Sie die Gesundheit von Schlüsselabhängigkeiten — Bibliotheken ohne Commits seit zwei Jahren, verlassene Maintainer oder rückläufige Community-Aktivität stellen ein Wartungsrisiko dar, das eine Ersatzsuche auslösen sollte, bevor eine Krise die Frage erzwingt.

## Tradeoffs ⇄

> Systematisches Abhängigkeitsmanagement verwandelt ein chronisches Hintergrundrisiko in einen handhabbaren, sichtbaren und auditierbaren Prozess, aber das Aufholen jahrelanger Vernachlässigung erfordert nachhaltige Investition, die mit Feature-Auslieferung konkurriert.

**Vorteile:**

- Schwachstellen-Scanning und automatisierte Update-Pull-Requests (Dependabot/Renovate) decken Sicherheitsrisiken auf, die sonst jahrelang tief im transitiven Abhängigkeitsbaum verborgen blieben — kritisch in Legacy-Systemen, wo Log4Shell-artige Schwachstellen unentdeckt lauern können.
- Lock-Dateien und deterministische Versionsauflösung eliminieren die „funktioniert auf meiner Maschine"-Probleme, die Legacy-Teams plagen, die ohne sie gebaut haben, was CI-Builds reproduzierbar und Debugging dramatisch leichter macht.
- Regelmäßige kleine Updates verhindern die Anhäufung von Upgrade-Schulden, die schließlich schmerzhafte mehrversionige Sprünge erzwingt; inkrementelle Updates sind handhabbar, während mehrjährige Lücken wochenlange Migrationsprojekte erzeugen.
- Eine aus dem verwalteten Abhängigkeitsbaum erzeugte Software Bill of Materials (SBOM) erfüllt regulatorische und unternehmerische Lieferketten-Anforderungen, die für Legacy-Systeme in regulierten Branchen zunehmend verpflichtend sind.
- Nicht mehr gepflegte Abhängigkeiten stillzulegen, bevor sie kritisch werden, erzwingt frühe architektonische Gespräche über Ersatz und gibt Teams Kontrolle über den Zeitplan, statt auf End-of-Life-Ankündigungen zu reagieren.

**Kosten und Risiken:**

- Die anfängliche Bereinigung eines lange vernachlässigten Abhängigkeitsbaums in einem großen Legacy-System kann Wochen an Aufwand erfordern, um Schwachstellen zu triagieren, Konflikte zu lösen und zu verifizieren, dass aktualisierte Bibliotheken das Verhalten in kritischen Codepfaden nicht ändern.
- Automatisierte Update-Pull-Requests von Dependabot erzeugen Falsch-Positive und Rauschen niedriger Schwere, das Teams überwältigen kann, wenn keine Triage-Richtlinie vorhanden ist, was zu Alarmmüdigkeit und ignorierten Updates führt.
- Legacy-Systeme, die ohne automatisierte Testabdeckung gebaut wurden, machen Abhängigkeitsupdates riskant — ohne eine Testsuite gibt es keinen zuverlässigen Weg zu verifizieren, dass eine aktualisierte Bibliothek das Verhalten nicht auf eine Weise geändert hat, die die Anwendung bricht.
- Major-Version-Upgrades grundlegender Frameworks (Spring Boot 2 auf 3, Angular 9 auf 17) in Legacy-Systemen können Änderungen über Hunderte von Dateien erfordern und stellen eigenständige Projekte dar, nicht bloß Abhängigkeitserhöhungen.
- Die Abhängigkeit von externen Maintainern, deren Prioritäten nicht zu den Bedürfnissen des Legacy-Systems passen, ist ein nicht reduzierbares Risiko; eine populäre Bibliothek, die den Support für eine ältere Java-Version einstellt, kann eine Upgrade-Kaskade erzwingen, auf die das Team nicht vorbereitet ist.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Abhängigkeitsmanagement aussieht, wenn es auf echte Legacy-System-Bedingungen angewandt wird.

Eine 2014 gebaute Java-Unternehmensanwendung lief 2023 immer noch mit Spring 4 und Hibernate 4. Das Team hatte Updates vermieden, weil der letzte größere Update-Versuch eine dreiwöchige Integrationskrise verursacht hatte. Als ein Sicherheitsaudit siebzehn hochschwere CVEs im transitiven Abhängigkeitsbaum enthüllte — mehrere in Frameworks, von denen das Team nicht einmal wusste, dass die Anwendung von ihnen abhing — erhielt das Projekt endlich Budget für Abhilfe. Das Team führte OWASP Dependency-Check in die CI-Pipeline ein, etablierte eine schriftliche Richtlinie für Sicherheitspatch-Zeitpläne und arbeitete den Upgrade-Rückstau über vier Sprints eine Bibliothek nach der anderen ab. Allein das Spring-Upgrade erforderte einen dedizierten zweiwöchigen Aufwand, aber der gestufte Ansatz verhinderte, dass sich das Projekt in eine weitere Integrationskrise verwandelte.

Ein Einzelhandelsunternehmen, das ein Node.js-Backend betrieb, entdeckte nach dem ersten Ausführen von `npm ls`, dass ihre Anwendung 1.400 transitive Abhängigkeiten hatte, von denen viele auf vor 2019 veröffentlichte Versionen fixiert waren. Die Aktivierung von Dependabot erzeugte eine sofortige Flut von über 200 Pull Requests. Das Team etablierte einen Triage-Prozess: kritische CVEs wurden sofort einem Entwickler zugewiesen, Minor-Version-Erhöhungen wurden wöchentlich gebündelt, und Major-Version-Erhöhungen wurden monatlich geplant. Innerhalb von drei Monaten war der Abhängigkeitsbaum aktuell, und der wöchentliche Batch war auf ein handhabbares Dutzend Routine-Updates geschrumpft.

Ein Regierungsauftragnehmer übernahm eine Python-Datenverarbeitungspipeline, die keine Requirements-Lock-Datei hatte — nur eine `requirements.txt` mit nicht fixierten Versionsbereichen. Die Pipeline hatte jahrelang auf einem spezifischen Server funktioniert, auf dem die installierten Paketversionen zufällig kompatibel waren, aber ein neuer Entwickler, der versuchte, sie lokal auszuführen, stellte fest, dass sie wegen inkompatibler transitiver Abhängigkeiten nicht startete. Die Einführung von `pip-tools`, um eine `requirements.txt.lock` zu erzeugen, sie ins Repository zu committen und die fixierten Versionen mit Snyk zu scannen, enthüllte zwei kritische Schwachstellen in fixierten transitiven Paketen, die über achtzehn Monate lang still vorhanden gewesen waren.
