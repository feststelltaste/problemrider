---
title: Wissenssilos
description: Wichtige Rechercheergebnisse und Expertise bleiben bei einzelnen Teammitgliedern
  isoliert, was Wissensaustausch und Teamlernen verhindert.
category:
- Culture
- Process
- Team
related_problems:
- slug: knowledge-dependency
  similarity: 0.75
- slug: team-silos
  similarity: 0.75
- slug: knowledge-sharing-breakdown
  similarity: 0.75
- slug: knowledge-gaps
  similarity: 0.75
- slug: single-points-of-failure
  similarity: 0.7
- slug: information-fragmentation
  similarity: 0.7
solutions:
- knowledge-sharing-practices
- pair-and-mob-programming
- architecture-documentation
- architecture-workshops
- code-reviews
- documentation-of-compatibility-requirements
- incident-management
- living-documentation
- on-call-duty
- security-community
- collaborative-problem-solving
- fair-source
- knowledge-base
- runbooks
- user-communities
- knowledge-rotation
- code-reading-sessions
- communities-of-practice
- system-decommissioning
- written-first-communication
- risk-quantification
- explicit-extension-points
layout: problem
lang: de
en_slug: knowledge-silos
---

## Description

Wissenssilos treten auf, wenn kritische Informationen, Expertise oder Rechercheergebnisse bei einzelnen Teammitgliedern konzentriert sind und nicht wirksam mit dem breiteren Team geteilt werden. Dies schafft Abhängigkeiten von bestimmten Personen, erhöht das Risiko, wenn Teammitglieder das Unternehmen verlassen, und führt zu dupliziertem Aufwand, während andere dieselben Informationen unabhängig neu entdecken müssen. Wissenssilos hindern Teams daran, kollektive Intelligenz aufzubauen und aus den Erfahrungen der anderen zu lernen.

## Indicators ⟡

- Bestimmte Teammitglieder sind durchgängig die "Ansprechperson" für spezifische Themen
- Informationen existieren, sind aber für andere Teammitglieder, die sie brauchen, nicht zugänglich
- Ähnliche Probleme werden von unterschiedlichen Teammitgliedern unterschiedlich gelöst
- Teamdiskussionen zeigen, dass Mitglieder unterschiedliche Verständnisse derselben Systeme haben
- Wissen geht verloren, wenn Schlüssel-Teammitglieder das Unternehmen verlassen oder nicht verfügbar sind

## Symptoms ▲

- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Wenn Wissen isoliert ist, werden Teammitglieder abhängig von den bestimmten Personen, die es besitzen.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  Isoliertes Wissen schafft Single Points of Failure, wenn Schlüsselpersonen nicht verfügbar sind oder gehen.
- [Doppelter Recherche-Aufwand](doppelter-recherche-aufwand.md)
<br/>  Ohne gemeinsames Wissen recherchieren mehrere Teammitglieder unabhängig dieselben Themen.
- [Verringerte Teamflexibilität](verringerte-teamflexibilitaet.md)
<br/>  Teammitglieder können nur in Bereichen arbeiten, in denen sie Wissen besitzen, was die Fähigkeit zur Arbeitsumverteilung verringert.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Übergreifende Probleme werden langsam gelöst, wenn jede Person nur ihre eigene Domäne versteht.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Wenn Wissen bei bestimmten Personen isoliert ist, können neue Entwickler nicht auf die Informationen zugreifen, die sie zum Einarbeiten brauchen.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Wissenssilos schaffen Engpässe, die die Teamproduktivität verringern.

## Causes ▼

- [Zusammenbruch des Wissensaustauschs](zusammenbruch-des-wissensaustauschs.md)
<br/>  Unwirksame Austauschmechanismen erlauben es Wissen, isoliert zu bleiben, statt verteilt zu werden.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Wenn kritisches Wissen nie formalisiert oder dokumentiert wird, bleibt es in den Köpfen von Einzelpersonen gefangen.
- [Implizites Erfahrungswissen](implizites-erfahrungswissen.md)
<br/>  Durch Erfahrung gewonnenes Wissen, das schwer zu artikulieren ist, schafft natürlich Silos.
- [Team-Silos](team-silos.md)
<br/>  Organisatorische Teamgrenzen verstärken Wissenssilos, indem sie teamübergreifende Interaktion einschränken.

## Detection Methods ○

- **Wissens-Mapping:** Identifikation, wer kritische Informationen über unterschiedliche Systembereiche besitzt
- **Informationsfluss-Analyse:** Nachverfolgung, wie Informationen durch das Team fließen (oder nicht fließen)
- **Bus-Faktor-Bewertung:** Bewertung des Risikos, falls bestimmte Teammitglieder nicht verfügbar werden
- **Teambefragungen:** Befragung zu Zugang zu Informationen und Erfahrungen mit Wissensaustausch
- **Dokumentations-Audit:** Überprüfung, welche kritischen Informationen dokumentiert vs. bei Einzelpersonen gehalten werden

## Examples

Ein Senior-Entwickler hat Monate damit verbracht, die Feinheiten des Zahlungsverarbeitungssystems zu lernen, einschließlich undokumentierter API-Eigenheiten, Fehlerbehandlungsmuster und Performance-Optimierungstechniken. Dieses Wissen bleibt in seinem Kopf und persönlichen Notizen, sodass, wenn er in den Urlaub geht, zahlungsbezogene Probleme viel länger brauchen, um gelöst zu werden, und neue Features verzögert werden. Andere Teammitglieder müssen dieselben Informationen durch Versuch und Irrtum neu entdecken. Ein weiteres Beispiel betrifft ein Team, in dem jeder Entwickler zum Experten für unterschiedliche Microservices geworden ist, sie aber ihr Verständnis von Serviceinteraktionen, Deployment-Verfahren oder Fehlerbehebungsansätzen nicht teilen. Wenn serviceübergreifende Probleme auftreten, versteht jeder Entwickler nur seinen Teil des Systems, was systemweite Probleme schwer zu diagnostizieren und zu lösen macht.
