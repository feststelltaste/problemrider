---
title: Sichere Programmierschnittstellen
description: Nutzung von Bibliotheken und Frameworks mit
  Sicherheitsfunktionen.
category:
- Security
- Code
problems:
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- obsolete-technologies
- technology-lock-in
- inadequate-error-handling
- dependency-version-conflicts
layout: solution
lang: de
en_slug: secure-programming-interfaces
related_solutions:
- slug: secure-coding-guidelines
  similarity: 0.8
- slug: security-frameworks
  similarity: 0.75
- slug: secure-by-default
  similarity: 0.75
- slug: security-tests
  similarity: 0.75
- slug: secure-software-development
  similarity: 0.75
- slug: secure-protocols
  similarity: 0.75
---

## Description

Die Nutzung sicherer Programmierschnittstellen bedeutet, sich auf die bereits in aktuelle Bibliotheken und Frameworks eingebauten Sicherheitsfeatures zu verlassen — Input-Sanitisierung, CSRF-Schutz, sichere Session-Behandlung, automatisches Output-Escaping —, statt benutzerdefinierte, handgestrickte Äquivalente derselben Funktionalität zu implementieren und zu pflegen. Da diese Bibliotheksfeatures von einer weit größeren Population von Entwicklern und Sicherheitsforschern genutzt und geprüft werden, als es der benutzerdefinierte Code eines einzelnen Teams je könnte, neigen sie dazu, Grenzfälle korrekt zu handhaben, die selbst gebaute Implementierungen übersehen, einfach durch die angesammelte Exposition weitverbreiteter Nutzung. Dies zählt in Legacy-Systemen besonders häufig, weil viele von ihnen die Verfügbarkeit dieser ausgereiften Bibliotheksfeatures überhaupt vordatieren, was bedeutet, dass ihr sicherheitsrelevanter Code — eine HTML-Escaping-Funktion, ein Session-Token-Generator — von Grund auf geschrieben wurde, zu einer Zeit, als der sicherere Standardansatz schlicht noch nicht als Fertiglösung existierte. Die Migration von solchem benutzerdefiniertem Code zum eingebauten Äquivalent eines modernen Frameworks schließt sowohl Lücken, die die ursprüngliche Implementierung nie berücksichtigte, als auch entfernt einen Bestand sicherheitskritischen Codes, den das Team sonst unbegrenzt weiter pflegen und erneut auditieren müsste. Die Migration ist nicht frei von eigenem Risiko, da das Upgrade eines Legacy-Frameworks, um diese Features zu erhalten, an anderer Stelle brechende Änderungen einführen kann, und jeder Wechsel weg von benutzerdefiniertem Sicherheitscode sorgfältig validiert werden muss, um zu bestätigen, dass der Ersatz echt äquivalenten — oder besseren — Schutz bietet, bevor die alte Implementierung entfernt wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Bewerten Sie aktuelle Bibliotheken und Frameworks auf eingebaute Sicherheitsfeatures wie Input-Sanitisierung, CSRF-Schutz und sichere Session-Behandlung
- Ersetzen Sie benutzerdefinierte Sicherheitsimplementierungen durch gut getestete Bibliotheksfunktionen, wo verfügbar
- Rüsten Sie Legacy-Frameworks auf Versionen auf, die moderne Sicherheitsfeatures standardmäßig einschließen
- Konfigurieren Sie Framework-Sicherheitsfeatures so, dass sie standardmäßig aktiviert sind, statt Opt-in zu erfordern
- Etablieren Sie eine Liste genehmigter Bibliotheken und Frameworks, die Sicherheitsanforderungen erfüllen
- Entfernen oder ersetzen Sie Bibliotheken mit bekannten ungepatchten Schwachstellen oder erreichtem Lebensende

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Nutzt community-getestete Sicherheitsimplementierungen statt benutzerdefinierten Codes
- Reduziert die Wahrscheinlichkeit häufiger Schwachstellen durch eingebaute Schutzmaßnahmen
- Hält Sicherheitsfähigkeiten durch Bibliotheks- und Framework-Updates aktuell
- Verringert die Menge sicherheitsspezifischen Codes, den das Team pflegen muss

**Kosten und Risiken:**
- Das Upgrade von Frameworks in Legacy-Systemen kann brechende Änderungen einführen
- Sich auf Framework-Standardeinstellungen zu verlassen erfordert Verständnis dessen, was diese Standardeinstellungen tatsächlich tun
- Bibliotheksschwachstellen können alle von ihnen abhängigen Anwendungen betreffen
- Die Migration von benutzerdefiniertem Sicherheitscode zu Framework-Features erfordert sorgfältiges Testen, um äquivalenten Schutz sicherzustellen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Die Legacy-Python-Webanwendung eines Medienunternehmens nutzte selbst gebaute HTML-Escaping-Funktionen, die 2010 geschrieben worden waren. Ein Sicherheitsreview fand, dass diese Funktionen mehrere Grenzfälle übersahen, die moderne Templating-Engines automatisch handhaben. Das Team migrierte von rohem String-Rendering zu Jinja2 mit aktiviertem Auto-Escaping, was eine gesamte Klasse von XSS-Schwachstellen beseitigte. Die Migration entfernte außerdem ungefähr 800 Zeilen benutzerdefinierten Sicherheitscodes, der keine Pflege mehr benötigte.
